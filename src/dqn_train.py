
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from alns import ALNSRunner
from dqn_policy import DoubleDQNPolicy, MLPQNetwork, default_dqn_actions, encode_state, feature_names
from dqn_replay_buffer import ReplayBuffer, ReplayTransition
from entropy import EntropyFeatureExtractor, EntropyFeatures
from experiment_runner import (
    build_optional_quantum_portfolio,
    load_noise_model,
    parse_int_list,
    predicted_context_noise_score,
    save_csv,
    stable_int_hash,
)
from pdptw import generate_pdptw_instance
from policy import EntropyAwareRepairOperator
from repair import GreedyPairRepair, RegretKPairRepair, build_classical_destroy_pool
from rl_training import (
    BANDIT_FEATURE_NAMES,
    RepairAction,
    RidgeActionValueModel,
    default_actions,
    load_bandit_policy,
    load_csv,
    train_policy_model,
)


class EpisodeTransitionCollector:
    def __init__(self) -> None:
        self.pending: Optional[dict] = None
        self.completed: List[ReplayTransition] = []

    def observe(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        action_mask: np.ndarray,
    ) -> None:
        state_row = {
            "state": np.asarray(state, dtype=float),
            "action": int(action),
            "reward": float(reward),
            "done": bool(done),
            "action_mask": np.asarray(action_mask, dtype=float),
        }
        if self.pending is not None:
            self.completed.append(
                ReplayTransition(
                    state=self.pending["state"].tolist(),
                    action=int(self.pending["action"]),
                    reward=float(self.pending["reward"]),
                    next_state=state_row["state"].tolist(),
                    done=False,
                    action_mask=self.pending["action_mask"].tolist(),
                    next_action_mask=state_row["action_mask"].tolist(),
                )
            )
        self.pending = state_row
        if done:
            self.finalize_episode()

    def finalize_episode(self) -> None:
        if self.pending is None:
            return
        zero_state = np.zeros_like(self.pending["state"], dtype=float)
        zero_mask = np.zeros_like(self.pending["action_mask"], dtype=float)
        self.completed.append(
            ReplayTransition(
                state=self.pending["state"].tolist(),
                action=int(self.pending["action"]),
                reward=float(self.pending["reward"]),
                next_state=zero_state.tolist(),
                done=bool(self.pending["done"]),
                action_mask=self.pending["action_mask"].tolist(),
                next_action_mask=zero_mask.tolist(),
            )
        )
        self.pending = None


class DQNTrainingPolicyAdapter:
    def __init__(self, controller: DoubleDQNPolicy, *, epsilon: float, seed: int = 0) -> None:
        self.controller = controller
        self.exact_max_states = int(controller.exact_max_states)
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)
        self.last_action_index = 0
        self.last_state = np.zeros(len(feature_names()), dtype=float)
        self.last_mask = np.zeros(len(controller.action_specs), dtype=float)
        self.iteration_fraction = 0.0
        self.accepted_last_step = 0.0
        self.improved_last_step = 0.0
        self.temperature_fraction = 1.0

    def choose(self, features, *, search_space: int):
        self.last_state = encode_state(
            features,
            search_space=search_space,
            iteration_fraction=self.iteration_fraction,
            accepted_last_step=self.accepted_last_step,
            improved_last_step=self.improved_last_step,
            temperature_fraction=self.temperature_fraction,
        )
        self.last_mask = self.controller.available_action_mask(features, search_space=search_space)
        self.last_action_index = self.controller.choose_action_index(
            self.last_state,
            action_mask=self.last_mask,
            epsilon=self.epsilon,
            rng=self.rng,
        )
        return self.controller.action_specs[self.last_action_index].to_decision()


def _fit_state_normalization(buffer: ReplayBuffer) -> tuple[np.ndarray, np.ndarray]:
    states, _, _, next_states, _, _, _ = buffer.to_numpy()
    all_states = np.vstack([states, next_states])
    mean = np.mean(all_states, axis=0)
    std = np.std(all_states, axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    return mean, std


def _build_initial_policy(
    *,
    shots: int,
    qiskit_max_states: int,
    hidden_dim: int,
    exact_max_states: int,
    gamma: float,
    seed: int,
) -> DoubleDQNPolicy:
    actions = default_dqn_actions(shots=shots, qiskit_max_states=qiskit_max_states, include_exact=True)
    network = MLPQNetwork(len(feature_names()), hidden_dim, len(actions), seed=seed)
    return DoubleDQNPolicy(
        q_network=network,
        action_specs=actions,
        state_mean=np.zeros(len(feature_names()), dtype=float),
        state_std=np.ones(len(feature_names()), dtype=float),
        gamma=gamma,
        exact_max_states=exact_max_states,
    )


def collect_dqn_replay(
    *,
    out_replay: Path,
    episodes: int,
    seeds: Iterable[int],
    sizes: Iterable[int],
    k_vehicles: int,
    iterations: int,
    remove_counts: Iterable[int],
    candidate_caps: Iterable[int],
    samples_per_call: int,
    tw_tightness: float,
    capacity_slack: float,
    noise_score: float,
    hardware_noise_model_path: Path | None,
    use_qiskit_aer: bool,
    qiskit_shots: int,
    qiskit_max_states: int,
    exact_max_states: int,
    epsilon: float,
    replay_capacity: int,
    out_episode_csv: Path | None = None,
) -> Dict[str, object]:
    replay = ReplayBuffer(capacity=replay_capacity)
    destroy_pool = build_classical_destroy_pool()
    quantum_operator = build_optional_quantum_portfolio(
        use_qiskit_aer=use_qiskit_aer,
        qiskit_shots=qiskit_shots,
        qiskit_max_states=qiskit_max_states,
    )
    hardware_noise_model = load_noise_model(hardware_noise_model_path)
    base_policy = _build_initial_policy(
        shots=qiskit_shots,
        qiskit_max_states=qiskit_max_states,
        hidden_dim=64,
        exact_max_states=exact_max_states,
        gamma=0.99,
        seed=1234,
    )
    episode_rows: List[Dict[str, object]] = []

    seed_list = list(seeds)
    size_list = list(sizes)
    remove_list = list(remove_counts)
    candidate_list = list(candidate_caps)

    for episode_idx in range(int(episodes)):
        seed = int(seed_list[episode_idx % max(1, len(seed_list))])
        n_requests = int(size_list[episode_idx % max(1, len(size_list))])
        remove_count = int(remove_list[episode_idx % max(1, len(remove_list))])
        candidate_cap = int(candidate_list[episode_idx % max(1, len(candidate_list))])
        effective_noise_score = predicted_context_noise_score(
            hardware_noise_model=hardware_noise_model,
            fallback_noise_score=noise_score,
            remove_count=remove_count,
            candidate_cap=candidate_cap,
            qiskit_shots=qiskit_shots,
        )
        inst = generate_pdptw_instance(
            n_requests,
            k_vehicles,
            seed=seed + 1000 * episode_idx,
            tw_tightness=tw_tightness,
            capacity_slack=capacity_slack,
        )
        adapter = DQNTrainingPolicyAdapter(
            base_policy,
            epsilon=epsilon,
            seed=seed + 17 * episode_idx,
        )
        repair_operator = EntropyAwareRepairOperator(
            extractor=EntropyFeatureExtractor(search_space_reference=10_000),
            policy=adapter,
            quantum_operator=quantum_operator,
            search_state={"stagnation": 0.0},
            noise_state={"noise_score": float(effective_noise_score)},
            quantum_eval_ratio=1.0,
        )
        collector = EpisodeTransitionCollector()

        def on_step(payload: Dict[str, object]) -> None:
            adapter.iteration_fraction = float(payload.get("iteration_fraction", 0.0))
            adapter.accepted_last_step = float(payload.get("accepted", False))
            adapter.improved_last_step = float(payload.get("improved_best", False))
            adapter.temperature_fraction = float(payload.get("temperature_fraction", 1.0))
            collector.observe(
                state=adapter.last_state,
                action=adapter.last_action_index,
                reward=float(payload.get("dqn_reward", payload.get("reward", 0.0))),
                done=bool(payload.get("done", False)),
                action_mask=adapter.last_mask,
            )

        runner = ALNSRunner(destroy_pool, [GreedyPairRepair(), RegretKPairRepair(2), repair_operator])
        _, best_eval, trace = runner.run(
            inst,
            seed=seed + stable_int_hash(f"dqn-collect:{episode_idx}:{remove_count}:{candidate_cap}", modulo=9_973),
            iterations=iterations,
            remove_count=remove_count,
            candidate_cap=candidate_cap,
            samples_per_call=samples_per_call,
            method_name="dqn_collect",
            noise_score=effective_noise_score,
            step_callback=on_step,
        )
        replay.extend(collector.completed)
        episode_rows.append(
            {
                "episode": episode_idx,
                "seed": seed,
                "n_requests": n_requests,
                "remove_count": remove_count,
                "candidate_cap": candidate_cap,
                "final_objective": best_eval.objective,
                "quantum_calls": trace.quantum_calls[-1] if trace.quantum_calls else 0,
                "transitions_total": len(replay),
            }
        )

    replay.save(out_replay)
    if out_episode_csv is not None:
        save_csv(out_episode_csv, episode_rows)
    return {"replay_path": str(out_replay), "transitions": len(replay), "episodes": len(episode_rows)}


def train_double_dqn(
    *,
    replay_path: Path,
    model_path: Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    target_sync_every: int,
    hidden_dim: int,
    qiskit_shots: int,
    qiskit_max_states: int,
    exact_max_states: int,
    seed: int = 1234,
    metrics_csv: Path | None = None,
) -> Dict[str, object]:
    rng = random.Random(seed)
    np.random.seed(seed)
    replay = ReplayBuffer.load(replay_path)
    if len(replay) < 4:
        raise ValueError("Need at least 4 transitions to train Double-DQN.")

    mean, std = _fit_state_normalization(replay)
    actions = default_dqn_actions(shots=qiskit_shots, qiskit_max_states=qiskit_max_states, include_exact=True)
    online = DoubleDQNPolicy(
        q_network=MLPQNetwork(len(feature_names()), hidden_dim, len(actions), seed=seed),
        action_specs=actions,
        state_mean=mean,
        state_std=std,
        gamma=gamma,
        exact_max_states=exact_max_states,
    )
    target = DoubleDQNPolicy(
        q_network=online.q_network.copy(),
        action_specs=actions,
        state_mean=mean,
        state_std=std,
        gamma=gamma,
        exact_max_states=exact_max_states,
    )

    metrics: List[Dict[str, object]] = []
    for step_idx in range(1, int(steps) + 1):
        batch = replay.sample(batch_size, rng=rng)
        states = np.asarray([item.state for item in batch], dtype=float)
        next_states = np.asarray([item.next_state for item in batch], dtype=float)
        rewards = np.asarray([item.reward for item in batch], dtype=float)
        actions_idx = np.asarray([item.action for item in batch], dtype=int)
        dones = np.asarray([item.done for item in batch], dtype=bool)
        masks = np.asarray([item.action_mask for item in batch], dtype=float)
        next_masks = np.asarray([item.next_action_mask for item in batch], dtype=float)

        states_std = np.asarray([online._standardize_state(row) for row in states], dtype=float)
        next_states_std = np.asarray([online._standardize_state(row) for row in next_states], dtype=float)

        q_online = online.q_network.predict(states_std)
        q_next_online = online.q_network.predict(next_states_std)
        q_next_target = target.q_network.predict(next_states_std)

        q_target = q_online.copy()
        for row_idx in range(len(batch)):
            masked_online = q_next_online[row_idx].copy()
            masked_online[next_masks[row_idx] <= 0.0] = -1e18
            if bool(dones[row_idx]) or np.all(next_masks[row_idx] <= 0.0):
                bootstrap = 0.0
            else:
                next_action = int(np.argmax(masked_online))
                bootstrap = float(q_next_target[row_idx, next_action])
            q_target[row_idx, actions_idx[row_idx]] = float(rewards[row_idx]) + (0.0 if dones[row_idx] else gamma * bootstrap)
            q_target[row_idx, masks[row_idx] <= 0.0] = q_online[row_idx, masks[row_idx] <= 0.0]

        loss = online.q_network.train_batch(states_std, q_target, learning_rate=learning_rate)

        if step_idx % int(target_sync_every) == 0:
            target.q_network = online.q_network.copy()

        metrics.append(
            {
                "train_step": step_idx,
                "loss": loss,
                "replay_size": len(replay),
                "batch_size": min(len(replay), int(batch_size)),
            }
        )

    online.save(model_path)
    if metrics_csv is not None:
        save_csv(metrics_csv, metrics)
    return {"model_path": str(model_path), "steps": int(steps), "replay_size": len(replay), "final_loss": metrics[-1]["loss"]}


def _coerce_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(row: Mapping[str, object], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _features_from_row(row: Mapping[str, object]) -> EntropyFeatures:
    return EntropyFeatures(
        insertion_cost_entropy=_coerce_float(row, "insertion_cost_entropy"),
        conflict_entropy=_coerce_float(row, "conflict_entropy"),
        slack_entropy=_coerce_float(row, "slack_entropy"),
        load_slack_entropy=_coerce_float(row, "load_slack_entropy"),
        structural_entropy=_coerce_float(row, "structural_entropy"),
        feasible_candidate_ratio=_coerce_float(row, "feasible_candidate_ratio"),
        conflict_density=_coerce_float(row, "conflict_density"),
        search_space_log=_coerce_float(row, "search_space_log"),
        removed_count=_coerce_int(row, "removed_count"),
        stagnation=_coerce_float(row, "stagnation"),
        noise_score=_coerce_float(row, "noise_score"),
    )


def _context_key(row: Mapping[str, object]) -> tuple[str, int]:
    instance_tag = str(row.get("instance_tag", ""))
    context_idx = _coerce_int(row, "context_idx", _coerce_int(row, "iteration"))
    return instance_tag, context_idx


def split_dataset_by_context(
    *,
    data_csv: Path,
    train_csv: Path,
    test_csv: Path,
    train_ratio: float = 0.8,
    seed: int = 1234,
) -> Dict[str, object]:
    rows = load_csv(data_csv)
    if not rows:
        raise ValueError(f"No rows found in {data_csv}.")
    context_keys = sorted({_context_key(row) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(context_keys)
    n_train = max(1, min(len(context_keys) - 1 if len(context_keys) > 1 else 1, int(round(train_ratio * len(context_keys)))))
    train_keys = set(context_keys[:n_train])
    train_rows = [row for row in rows if _context_key(row) in train_keys]
    test_rows = [row for row in rows if _context_key(row) not in train_keys]
    if not test_rows:
        test_rows = train_rows[-1:]
        train_rows = train_rows[:-1] or train_rows
    save_csv(train_csv, train_rows)
    save_csv(test_csv, test_rows)
    return {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_contexts": len({_context_key(row) for row in train_rows}),
        "test_contexts": len({_context_key(row) for row in test_rows}),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }


def dataset_rows_to_replay(
    rows: Sequence[Mapping[str, object]],
    *,
    qiskit_shots: int,
    qiskit_max_states: int,
    exact_max_states: int,
    replay_capacity: int = 300_000,
) -> ReplayBuffer:
    action_lookup = {action.name: idx for idx, action in enumerate(default_dqn_actions(shots=qiskit_shots, qiskit_max_states=qiskit_max_states, include_exact=True))}
    replay = ReplayBuffer(capacity=replay_capacity)
    grouped: Dict[tuple[str, int], List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_context_key(row), []).append(row)
    for context_rows in grouped.values():
        sample_row = context_rows[0]
        features = _features_from_row(sample_row)
        search_space = _coerce_int(sample_row, "search_space", 0)
        state = encode_state(features, search_space=search_space)
        dqn_policy = _build_initial_policy(
            shots=qiskit_shots,
            qiskit_max_states=qiskit_max_states,
            hidden_dim=8,
            exact_max_states=exact_max_states,
            gamma=0.99,
            seed=0,
        )
        action_mask = dqn_policy.available_action_mask(features, search_space=search_space)
        valid_rows = [row for row in context_rows if str(row.get("action_name", "")) in action_lookup]
        if not valid_rows:
            continue
        best_row = max(valid_rows, key=lambda row: _coerce_float(row, "reward"))
        action_idx = action_lookup[str(best_row.get("action_name"))]
        replay.add(
            ReplayTransition(
                state=state.tolist(),
                action=int(action_idx),
                reward=_coerce_float(best_row, "reward"),
                next_state=np.zeros_like(state).tolist(),
                done=True,
                action_mask=action_mask.tolist(),
                next_action_mask=np.zeros_like(action_mask).tolist(),
            )
        )
    return replay


def train_double_dqn_from_dataset(
    *,
    data_csv: Path,
    model_path: Path,
    replay_path: Path,
    metrics_csv: Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    target_sync_every: int,
    hidden_dim: int,
    qiskit_shots: int,
    qiskit_max_states: int,
    exact_max_states: int,
    seed: int = 1234,
) -> Dict[str, object]:
    rows = load_csv(data_csv)
    replay = dataset_rows_to_replay(
        rows,
        qiskit_shots=qiskit_shots,
        qiskit_max_states=qiskit_max_states,
        exact_max_states=exact_max_states,
    )
    if len(replay) < 4:
        raise ValueError("Need at least 4 grouped contexts to train Double-DQN from dataset.")
    replay.save(replay_path)
    outputs = train_double_dqn(
        replay_path=replay_path,
        model_path=model_path,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        target_sync_every=target_sync_every,
        hidden_dim=hidden_dim,
        qiskit_shots=qiskit_shots,
        qiskit_max_states=qiskit_max_states,
        exact_max_states=exact_max_states,
        seed=seed,
        metrics_csv=metrics_csv,
    )
    outputs["dataset_rows"] = len(rows)
    outputs["dataset_contexts"] = len({_context_key(row) for row in rows})
    return outputs


def _group_rows_by_context(rows: Sequence[Mapping[str, object]]) -> Dict[tuple[str, int], List[Mapping[str, object]]]:
    grouped: Dict[tuple[str, int], List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_context_key(row), []).append(row)
    return grouped


def _pick_chosen_row_by_name(context_rows: Sequence[Mapping[str, object]], action_name: str) -> Optional[Mapping[str, object]]:
    exact = [row for row in context_rows if str(row.get("action_name")) == action_name]
    if exact:
        return max(exact, key=lambda row: _coerce_float(row, "reward"))
    return None


def evaluate_policies_from_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    dqn_policy: DoubleDQNPolicy | None,
    bandit_model_path: Path | None,
    exact_max_states: int,
    out_csv: Path,
    summary_csv: Path,
    split_name: str,
    min_quantum_observations: int = 0,
) -> Dict[str, object]:
    grouped = _group_rows_by_context(rows)
    bandit_policy = None
    if bandit_model_path is not None and bandit_model_path.exists():
        bandit_policy = load_bandit_policy(
            bandit_model_path,
            exact_max_states=exact_max_states,
            min_quantum_observations=min_quantum_observations,
        )

    eval_rows: List[Dict[str, object]] = []
    for context_key, context_rows in grouped.items():
        sample = context_rows[0]
        features = _features_from_row(sample)
        search_space = _coerce_int(sample, "search_space", 0)
        best_any = max(context_rows, key=lambda row: _coerce_float(row, "reward"))
        classical_rows = [row for row in context_rows if _coerce_int(row, "use_quantum", 0) == 0]
        quantum_rows = [row for row in context_rows if _coerce_int(row, "use_quantum", 0) == 1]
        best_classical = max(classical_rows, key=lambda row: _coerce_float(row, "reward")) if classical_rows else None
        best_quantum = max(quantum_rows, key=lambda row: _coerce_float(row, "reward")) if quantum_rows else None

        if dqn_policy is not None:
            state = encode_state(features, search_space=search_space)
            mask = dqn_policy.available_action_mask(features, search_space=search_space)
            action_idx = dqn_policy.choose_action_index(state, action_mask=mask, epsilon=0.0, rng=random.Random(0))
            dqn_action = dqn_policy.action_specs[action_idx].name
            dqn_row = _pick_chosen_row_by_name(context_rows, dqn_action)
        else:
            dqn_row = None

        bandit_row = None
        if bandit_policy is not None:
            decision = bandit_policy.choose(features, search_space=search_space)
            candidates = []
            for row in context_rows:
                repair_name = str(row.get("repair_name", ""))
                use_quantum = _coerce_int(row, "use_quantum", 0) == 1
                family = str(row.get("family", ""))
                layers = _coerce_int(row, "layers", 0)
                if repair_name != decision.repair_name:
                    continue
                if bool(use_quantum) != bool(decision.use_quantum):
                    continue
                if decision.use_quantum and decision.circuit is not None:
                    if family != decision.circuit.family or layers != int(decision.circuit.layers):
                        continue
                candidates.append(row)
            if candidates:
                bandit_row = max(candidates, key=lambda row: _coerce_float(row, "reward"))

        policies = {
            "oracle_best": best_any,
            "oracle_classical": best_classical,
            "oracle_quantum": best_quantum,
            "dqn_policy": dqn_row,
            "quantum_policy": bandit_row,
        }
        for policy_name, chosen in policies.items():
            if chosen is None:
                continue
            eval_rows.append(
                {
                    "split": split_name,
                    "instance_tag": context_key[0],
                    "context_idx": context_key[1],
                    "policy": policy_name,
                    "action_name": str(chosen.get("action_name", "")),
                    "repair_name": str(chosen.get("repair_name", "")),
                    "use_quantum": _coerce_int(chosen, "use_quantum", 0),
                    "reward": _coerce_float(chosen, "reward"),
                    "raw_reward": _coerce_float(chosen, "raw_reward", _coerce_float(chosen, "reward")),
                    "baseline_reward": _coerce_float(chosen, "baseline_reward", 0.0),
                    "search_space": search_space,
                }
            )

    save_csv(out_csv, eval_rows)
    if not eval_rows:
        raise ValueError("No evaluation rows were generated.")
    summary_rows: List[Dict[str, object]] = []
    grouped_summary: Dict[tuple[str, str], List[Mapping[str, object]]] = {}
    for row in eval_rows:
        grouped_summary.setdefault((str(row["split"]), str(row["policy"])), []).append(row)
    for (split, policy), rows_for_policy in sorted(grouped_summary.items()):
        rewards = np.asarray([_coerce_float(row, "reward") for row in rows_for_policy], dtype=float)
        raw_rewards = np.asarray([_coerce_float(row, "raw_reward") for row in rows_for_policy], dtype=float)
        summary_rows.append(
            {
                "split": split,
                "policy": policy,
                "contexts": len(rows_for_policy),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "sum_reward": float(np.sum(rewards)),
                "mean_raw_reward": float(np.mean(raw_rewards)),
                "sum_raw_reward": float(np.sum(raw_rewards)),
                "quantum_rate": float(np.mean([_coerce_int(row, "use_quantum", 0) for row in rows_for_policy])),
            }
        )
    save_csv(summary_csv, summary_rows)
    return {"eval_rows": len(eval_rows), "contexts": len(grouped), "summary_csv": str(summary_csv), "out_csv": str(out_csv)}


def evaluate_policy_models(
    *,
    data_csv: Path,
    dqn_model_path: Path | None,
    bandit_model_path: Path | None,
    out_csv: Path,
    summary_csv: Path,
    split_name: str = "all",
    exact_max_states: int = 128,
    min_quantum_observations: int = 0,
) -> Dict[str, object]:
    rows = load_csv(data_csv)
    dqn_policy = DoubleDQNPolicy.load(dqn_model_path) if dqn_model_path is not None and dqn_model_path.exists() else None
    return evaluate_policies_from_rows(
        rows=rows,
        dqn_policy=dqn_policy,
        bandit_model_path=bandit_model_path,
        exact_max_states=exact_max_states,
        out_csv=out_csv,
        summary_csv=summary_csv,
        split_name=split_name,
        min_quantum_observations=min_quantum_observations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect replay, train Double-DQN, and compare against the quantum policy.")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Collect replay transitions from ALNS episodes.")
    collect.add_argument("--out-replay", type=Path, default=Path("runs/entropy_pdptw_dqn/dqn_replay.json"))
    collect.add_argument("--out-episode-csv", type=Path, default=Path("runs/entropy_pdptw_dqn/dqn_collect_episodes.csv"))
    collect.add_argument("--episodes", type=int, default=200)
    collect.add_argument("--seeds", type=str, default="1,2,3")
    collect.add_argument("--sizes", type=str, default="15,20")
    collect.add_argument("--k-vehicles", type=int, default=6)
    collect.add_argument("--iterations", type=int, default=80)
    collect.add_argument("--remove-counts", type=str, default="2,3,4,5")
    collect.add_argument("--candidate-caps", type=str, default="2,3,4")
    collect.add_argument("--samples-per-call", type=int, default=16)
    collect.add_argument("--tw-tightness", type=float, default=0.5)
    collect.add_argument("--capacity-slack", type=float, default=0.30)
    collect.add_argument("--noise-score", type=float, default=0.15)
    collect.add_argument("--hardware-noise-model", type=Path, default=None)
    collect.add_argument("--use-qiskit-aer", action="store_true")
    collect.add_argument("--qiskit-shots", type=int, default=256)
    collect.add_argument("--qiskit-max-states", type=int, default=256)
    collect.add_argument("--exact-max-states", type=int, default=128)
    collect.add_argument("--epsilon", type=float, default=0.20)
    collect.add_argument("--replay-capacity", type=int, default=300_000)

    train = sub.add_parser("train", help="Train Double-DQN from a saved replay buffer.")
    train.add_argument("--replay-path", type=Path, default=Path("runs/entropy_pdptw_dqn/dqn_replay.json"))
    train.add_argument("--model-path", type=Path, default=Path("runs/entropy_pdptw_dqn/dqn_policy.json"))
    train.add_argument("--metrics-csv", type=Path, default=Path("runs/entropy_pdptw_dqn/dqn_train_metrics.csv"))
    train.add_argument("--steps", type=int, default=1_000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--target-sync-every", type=int, default=50)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--qiskit-shots", type=int, default=256)
    train.add_argument("--qiskit-max-states", type=int, default=256)
    train.add_argument("--exact-max-states", type=int, default=128)
    train.add_argument("--seed", type=int, default=1234)

    train_dataset = sub.add_parser("train-dataset", help="Train Double-DQN directly from an action dataset.")
    train_dataset.add_argument("--data-csv", type=Path, default=Path("runs/entropy_pdptw/training_actions.csv"))
    train_dataset.add_argument("--model-path", type=Path, default=Path("runs/entropy_pdptw_dqn/dqn_policy.json"))
    train_dataset.add_argument("--replay-path", type=Path, default=Path("runs/entropy_pdptw_dqn/dataset_replay.json"))
    train_dataset.add_argument("--metrics-csv", type=Path, default=Path("runs/entropy_pdptw_dqn/dqn_train_metrics.csv"))
    train_dataset.add_argument("--steps", type=int, default=1_000)
    train_dataset.add_argument("--batch-size", type=int, default=64)
    train_dataset.add_argument("--learning-rate", type=float, default=1e-3)
    train_dataset.add_argument("--gamma", type=float, default=0.99)
    train_dataset.add_argument("--target-sync-every", type=int, default=50)
    train_dataset.add_argument("--hidden-dim", type=int, default=64)
    train_dataset.add_argument("--qiskit-shots", type=int, default=256)
    train_dataset.add_argument("--qiskit-max-states", type=int, default=256)
    train_dataset.add_argument("--exact-max-states", type=int, default=128)
    train_dataset.add_argument("--seed", type=int, default=1234)

    split = sub.add_parser("split-dataset", help="Create a train/test split grouped by context.")
    split.add_argument("--data-csv", type=Path, default=Path("runs/entropy_pdptw/training_actions.csv"))
    split.add_argument("--train-csv", type=Path, default=Path("runs/entropy_pdptw/train_actions.csv"))
    split.add_argument("--test-csv", type=Path, default=Path("runs/entropy_pdptw/test_actions.csv"))
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--seed", type=int, default=1234)

    evaluate = sub.add_parser("evaluate", help="Evaluate offline policy reward on a dataset split.")
    evaluate.add_argument("--data-csv", type=Path, required=True)
    evaluate.add_argument("--dqn-model-path", type=Path, default=None)
    evaluate.add_argument("--bandit-model-path", type=Path, default=None)
    evaluate.add_argument("--out-csv", type=Path, default=Path("runs/entropy_pdptw_eval/policy_reward_eval.csv"))
    evaluate.add_argument("--summary-csv", type=Path, default=Path("runs/entropy_pdptw_eval/policy_reward_summary.csv"))
    evaluate.add_argument("--split-name", type=str, default="all")
    evaluate.add_argument("--exact-max-states", type=int, default=128)
    evaluate.add_argument("--min-quantum-observations", type=int, default=0)

    args = parser.parse_args()

    if args.command == "collect":
        outputs = collect_dqn_replay(
            out_replay=args.out_replay,
            episodes=args.episodes,
            seeds=parse_int_list(args.seeds),
            sizes=parse_int_list(args.sizes),
            k_vehicles=args.k_vehicles,
            iterations=args.iterations,
            remove_counts=parse_int_list(args.remove_counts),
            candidate_caps=parse_int_list(args.candidate_caps),
            samples_per_call=args.samples_per_call,
            tw_tightness=args.tw_tightness,
            capacity_slack=args.capacity_slack,
            noise_score=args.noise_score,
            hardware_noise_model_path=args.hardware_noise_model,
            use_qiskit_aer=args.use_qiskit_aer,
            qiskit_shots=args.qiskit_shots,
            qiskit_max_states=args.qiskit_max_states,
            exact_max_states=args.exact_max_states,
            epsilon=args.epsilon,
            replay_capacity=args.replay_capacity,
            out_episode_csv=args.out_episode_csv,
        )
    elif args.command == "train":
        outputs = train_double_dqn(
            replay_path=args.replay_path,
            model_path=args.model_path,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            target_sync_every=args.target_sync_every,
            hidden_dim=args.hidden_dim,
            qiskit_shots=args.qiskit_shots,
            qiskit_max_states=args.qiskit_max_states,
            exact_max_states=args.exact_max_states,
            seed=args.seed,
            metrics_csv=args.metrics_csv,
        )
    elif args.command == "train-dataset":
        outputs = train_double_dqn_from_dataset(
            data_csv=args.data_csv,
            model_path=args.model_path,
            replay_path=args.replay_path,
            metrics_csv=args.metrics_csv,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            target_sync_every=args.target_sync_every,
            hidden_dim=args.hidden_dim,
            qiskit_shots=args.qiskit_shots,
            qiskit_max_states=args.qiskit_max_states,
            exact_max_states=args.exact_max_states,
            seed=args.seed,
        )
    elif args.command == "split-dataset":
        outputs = split_dataset_by_context(
            data_csv=args.data_csv,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    else:
        outputs = evaluate_policy_models(
            data_csv=args.data_csv,
            dqn_model_path=args.dqn_model_path,
            bandit_model_path=args.bandit_model_path,
            out_csv=args.out_csv,
            summary_csv=args.summary_csv,
            split_name=args.split_name,
            exact_max_states=args.exact_max_states,
            min_quantum_observations=args.min_quantum_observations,
        )

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
