from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from hardware_noise import CircuitNoiseContext, EmpiricalHardwareNoiseModel
except Exception:  # pragma: no cover - optional helper
    CircuitNoiseContext = None
    EmpiricalHardwareNoiseModel = None

from entropy import EntropyFeatureExtractor, EntropyFeatures
from pdptw import PDPTWInstance, evaluate_solution, generate_pdptw_instance, greedy_initial_solution
from policy import CircuitConfig, QuantumRepairPortfolio, RepairDecision
from repair import (
    EntropyBiasedSamplerRepair,
    ExactMicroSearchRepair,
    GreedyPairRepair,
    RegretKPairRepair,
    RepairContext,
    RepairOperator,
    RepairProposal,
    build_classical_destroy_pool,
    build_quantum_repair_operator,
    build_repair_context,
)


BANDIT_FEATURE_NAMES = [
    "insertion_cost_entropy",
    "conflict_entropy",
    "slack_entropy",
    "load_slack_entropy",
    "structural_entropy",
    "feasible_candidate_ratio",
    "conflict_density",
    "search_space_log",
    "removed_count",
    "stagnation",
    "noise_score",
    "search_space_raw_log",
    "bias_context",
]


@dataclass(frozen=True)
class RepairAction:
    """One discrete action available to the contextual bandit."""

    name: str
    repair_name: str
    use_quantum: bool = False
    family: str = ""
    layers: int = 0
    shots: int = 0
    target_entropy: float = 0.0
    max_states: int = 512

    def to_decision(self) -> RepairDecision:
        if not self.use_quantum:
            return RepairDecision(self.repair_name, False, None, "rl contextual-bandit classical action")

        circuit = CircuitConfig(
            family=self.family,
            layers=int(self.layers),
            shots=int(self.shots),
            target_entropy=float(self.target_entropy),
        )
        return RepairDecision("quantum_repair", True, circuit, "rl contextual-bandit quantum action")


@dataclass
class ActionEvaluation:
    action: RepairAction
    proposal: RepairProposal
    reward: float
    runtime_sec: float
    action_noise_score: float = 0.0
    failed: bool = False
    error: str = ""


class RidgeActionValueModel:
    """One linear ridge reward model per action."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        action_specs: Sequence[RepairAction],
        ridge_alpha: float = 1.0,
        quantum_margin: float = 0.02,
    ) -> None:
        self.feature_names = list(feature_names)
        self.action_specs = {action.name: action for action in action_specs}
        self.ridge_alpha = float(ridge_alpha)
        self.quantum_margin = float(quantum_margin)
        self.feature_mean: List[float] = []
        self.feature_std: List[float] = []
        self.coefficients: Dict[str, List[float]] = {}
        self.action_counts: Dict[str, int] = {}

    def _matrix_from_rows(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        values = []
        for row in rows:
            values.append([float(row.get(name, 0.0)) for name in self.feature_names])
        return np.asarray(values, dtype=float)

    def _standardize_fit(self, x_raw: np.ndarray) -> np.ndarray:
        mean = np.mean(x_raw, axis=0)
        std = np.std(x_raw, axis=0)
        std = np.where(std <= 1e-12, 1.0, std)
        self.feature_mean = mean.tolist()
        self.feature_std = std.tolist()
        return (x_raw - mean) / std

    def _standardize_predict(self, x_raw: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.feature_mean, dtype=float)
        std = np.asarray(self.feature_std, dtype=float)
        std = np.where(std <= 1e-12, 1.0, std)
        return (x_raw - mean) / std

    def fit(self, rows: Sequence[Mapping[str, object]]) -> None:
        if not rows:
            raise ValueError("Cannot train contextual bandit on an empty dataset.")

        x_raw = self._matrix_from_rows(rows)
        x_std = self._standardize_fit(x_raw)
        x_design = np.column_stack([np.ones(len(x_std)), x_std])
        all_actions = sorted({str(row["action_name"]) for row in rows} | set(self.action_specs))

        self.coefficients = {}
        self.action_counts = {}

        for action_name in all_actions:
            indices = [idx for idx, row in enumerate(rows) if str(row["action_name"]) == action_name]
            self.action_counts[action_name] = len(indices)
            if not indices:
                self.coefficients[action_name] = [0.0] * x_design.shape[1]
                continue

            xa = x_design[indices]
            ya = np.asarray([float(rows[idx]["reward"]) for idx in indices], dtype=float)
            penalty = self.ridge_alpha * np.eye(xa.shape[1])
            penalty[0, 0] = 0.0
            coef = np.linalg.solve(xa.T @ xa + penalty, xa.T @ ya)
            self.coefficients[action_name] = coef.tolist()

    def predict_values(self, feature_dict: Mapping[str, float]) -> Dict[str, float]:
        x_raw = np.asarray([[float(feature_dict.get(name, 0.0)) for name in self.feature_names]], dtype=float)
        x_std = self._standardize_predict(x_raw)
        x_design = np.concatenate([np.ones((1, 1)), x_std], axis=1)
        out: Dict[str, float] = {}
        for action_name, coef_values in self.coefficients.items():
            coef = np.asarray(coef_values, dtype=float)
            out[action_name] = float((x_design @ coef)[0])
        return out

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "model_type": "ridge_action_value",
            "feature_names": self.feature_names,
            "actions": [asdict(action) for action in self.action_specs.values()],
            "ridge_alpha": self.ridge_alpha,
            "quantum_margin": self.quantum_margin,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "coefficients": self.coefficients,
            "action_counts": self.action_counts,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "RidgeActionValueModel":
        actions = [RepairAction(**item) for item in payload.get("actions", [])]
        model = cls(
            feature_names=list(payload["feature_names"]),
            action_specs=actions,
            ridge_alpha=float(payload.get("ridge_alpha", 1.0)),
            quantum_margin=float(payload.get("quantum_margin", 0.02)),
        )
        model.feature_mean = [float(v) for v in payload["feature_mean"]]
        model.feature_std = [float(v) for v in payload["feature_std"]]
        model.coefficients = {str(k): [float(x) for x in v] for k, v in dict(payload["coefficients"]).items()}
        model.action_counts = {str(k): int(v) for k, v in dict(payload.get("action_counts", {})).items()}
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RidgeActionValueModel":
        return cls.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


class HierarchicalBanditPolicy:
    """Deployable policy with a quantum gate and a circuit selector."""

    def __init__(
        self,
        model: RidgeActionValueModel,
        *,
        exact_max_states: int = 128,
        quantum_margin: Optional[float] = None,
        hard_quantum_noise_cap: float = 0.85,
        min_quantum_observations: int = 5,
    ) -> None:
        self.model = model
        self.exact_max_states = int(exact_max_states)
        self.quantum_margin = model.quantum_margin if quantum_margin is None else float(quantum_margin)
        self.hard_quantum_noise_cap = float(hard_quantum_noise_cap)
        self.min_quantum_observations = int(min_quantum_observations)

    def _available_actions(self, features: EntropyFeatures, search_space: int) -> List[RepairAction]:
        actions = []
        for action in self.model.action_specs.values():
            if action.repair_name == "exact_micro_repair" and search_space > self.exact_max_states:
                continue
            if action.use_quantum:
                if search_space > action.max_states:
                    continue
                if features.noise_score > self.hard_quantum_noise_cap:
                    continue
            actions.append(action)
        return actions

    def choose(self, features: EntropyFeatures, *, search_space: int) -> RepairDecision:
        feature_dict = features_to_bandit_dict(features, search_space=search_space)
        values = self.model.predict_values(feature_dict)
        available = self._available_actions(features, search_space)
        if not available:
            return RepairDecision("regret2_repair", False, None, "rl fallback: no available action")

        classical = [action for action in available if not action.use_quantum]
        quantum = [
            action
            for action in available
            if action.use_quantum
            and self.model.action_counts.get(action.name, 0) >= self.min_quantum_observations
        ]

        best_classical = max(classical, key=lambda action: values.get(action.name, -1e18)) if classical else None
        best_quantum = max(quantum, key=lambda action: values.get(action.name, -1e18)) if quantum else None

        if best_quantum is not None:
            quantum_value = values.get(best_quantum.name, -1e18)
            classical_value = values.get(best_classical.name, -1e18) if best_classical is not None else -1e18
            if quantum_value > classical_value + self.quantum_margin:
                return best_quantum.to_decision()

        if best_classical is not None:
            return best_classical.to_decision()

        return RepairDecision("regret2_repair", False, None, "rl fallback: no classical action")


def stable_int_hash(value: str, *, modulo: int = 2_147_483_647) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(modulo)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def features_to_bandit_dict(features: EntropyFeatures, *, search_space: int) -> Dict[str, float]:
    values = features.as_dict()
    values["search_space_raw_log"] = math.log1p(max(0, int(search_space)))
    values["bias_context"] = 1.0
    return {name: float(values.get(name, 0.0)) for name in BANDIT_FEATURE_NAMES}


def default_actions(*, shots: int = 256, qiskit_max_states: int = 256, include_exact: bool = True) -> List[RepairAction]:
    actions = [
        RepairAction("classical:greedy", "greedy_repair"),
        RepairAction("classical:regret2", "regret2_repair"),
        RepairAction("classical:regret3", "regret3_repair"),
    ]
    if include_exact:
        actions.append(RepairAction("classical:exact_micro", "exact_micro_repair"))
    actions.extend(
        [
            RepairAction(
                f"quantum:qaoa:L1:S{shots}:T0.45",
                "quantum_repair",
                use_quantum=True,
                family="qaoa",
                layers=1,
                shots=shots,
                target_entropy=0.45,
                max_states=qiskit_max_states,
            ),
            RepairAction(
                f"quantum:qaoa:L2:S{shots}:T0.50",
                "quantum_repair",
                use_quantum=True,
                family="qaoa",
                layers=2,
                shots=shots,
                target_entropy=0.50,
                max_states=qiskit_max_states,
            ),
            RepairAction(
                f"quantum:vqa_su2:L1:S{shots}:T0.60",
                "quantum_repair",
                use_quantum=True,
                family="vqa_su2",
                layers=1,
                shots=shots,
                target_entropy=0.60,
                max_states=qiskit_max_states,
            ),
            RepairAction(
                f"quantum:vqa_su2:L2:S{shots}:T0.70",
                "quantum_repair",
                use_quantum=True,
                family="vqa_su2",
                layers=2,
                shots=shots,
                target_entropy=0.70,
                max_states=qiskit_max_states,
            ),
        ]
    )
    return actions


def build_action_operator(
    action: RepairAction,
    *,
    quantum_portfolio: Optional[QuantumRepairPortfolio],
    exact_max_states: int,
) -> RepairOperator:
    if action.repair_name == "greedy_repair":
        return GreedyPairRepair()
    if action.repair_name == "regret2_repair":
        return RegretKPairRepair(2)
    if action.repair_name == "regret3_repair":
        return RegretKPairRepair(3)
    if action.repair_name == "exact_micro_repair":
        return ExactMicroSearchRepair(max_states=exact_max_states)
    if action.use_quantum:
        if quantum_portfolio is None:
            return EntropyBiasedSamplerRepair(max_states=action.max_states, beta=1.0)
        return quantum_portfolio.select(action.to_decision().circuit)
    raise ValueError(f"Unknown repair action: {action}")


def repair_reward(
    *,
    proposal: RepairProposal,
    ctx: RepairContext,
    runtime_sec: float,
    action: RepairAction,
    noise_score: float,
    feasible_bonus: float = 1.0,
    infeasible_penalty: float = 2.0,
    runtime_penalty: float = 0.05,
    quantum_noise_penalty: float = 0.25,
) -> float:
    scale = max(1.0, abs(float(ctx.base_eval.objective)))
    delta_score = -float(proposal.exact_delta) / scale
    feasibility_score = feasible_bonus if proposal.feasible else -infeasible_penalty
    noise_penalty = quantum_noise_penalty * float(noise_score) if action.use_quantum else 0.0
    return float(delta_score + feasibility_score - runtime_penalty * float(runtime_sec) - noise_penalty)


def evaluate_action(
    *,
    ctx: RepairContext,
    action: RepairAction,
    rng: random.Random,
    n_samples: int,
    quantum_portfolio: Optional[QuantumRepairPortfolio],
    exact_max_states: int,
    noise_score: float,
) -> ActionEvaluation:
    started_at = time.perf_counter()
    try:
        operator = build_action_operator(action, quantum_portfolio=quantum_portfolio, exact_max_states=exact_max_states)
        proposal = operator.best_repair(ctx, rng, n_samples)
        runtime_sec = time.perf_counter() - started_at
        reward = repair_reward(
            proposal=proposal,
            ctx=ctx,
            runtime_sec=runtime_sec,
            action=action,
            noise_score=noise_score,
        )
        return ActionEvaluation(action, proposal, reward, runtime_sec, action_noise_score=float(noise_score))
    except Exception as exc:
        fallback = RegretKPairRepair(2).best_repair(ctx, rng, max(1, n_samples))
        runtime_sec = time.perf_counter() - started_at
        return ActionEvaluation(
            action=action,
            proposal=fallback,
            reward=-10.0,
            runtime_sec=runtime_sec,
            action_noise_score=float(noise_score),
            failed=True,
            error=str(exc),
        )


def make_quantum_portfolio(
    *,
    use_qiskit_aer: bool,
    shots: int,
    qiskit_max_states: int,
    transpile_seed: int = 1234,
) -> Optional[QuantumRepairPortfolio]:
    if not use_qiskit_aer:
        return None

    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise RuntimeError("Install qiskit-aer or collect training data without --use-qiskit-aer.") from exc

    backend = AerSimulator(seed_simulator=transpile_seed)
    operators: Dict[str, RepairOperator] = {}
    for family in ("qaoa", "vqa_su2"):
        for layers in (1, 2):
            operator = build_quantum_repair_operator(
                family=family,
                backend=backend,
                layers=layers,
                shots=shots,
                max_states=qiskit_max_states,
                transpile_seed=transpile_seed + 31 * layers + stable_int_hash(family, modulo=997),
            )
            operators[f"{family}:L{layers}:S{shots}"] = operator
            operators[f"{family}:L{layers}"] = operator
            operators.setdefault(family, operator)

    return QuantumRepairPortfolio(operators)


def load_training_noise_model(path: Optional[Path]):
    if path is None:
        return None
    if EmpiricalHardwareNoiseModel is None:
        raise RuntimeError("hardware_noise.py could not be imported, so --hardware-noise-model cannot be used.")
    return EmpiricalHardwareNoiseModel.load(Path(path))


def predict_action_noise_score(
    *,
    action: RepairAction,
    hardware_noise_model,
    fallback_noise_score: float,
    num_removed: int,
    candidate_cap: int,
    qiskit_shots: int,
) -> float:
    if not action.use_quantum:
        return 0.0
    if hardware_noise_model is None or CircuitNoiseContext is None:
        return float(fallback_noise_score)
    prediction = hardware_noise_model.predict(
        CircuitNoiseContext(
            family=action.family,
            layers=max(1, int(action.layers)),
            shots=int(action.shots or qiskit_shots),
            num_removed=int(num_removed),
            candidates_per_removed=int(candidate_cap),
        )
    )
    return float(prediction.noise_score)


def predict_context_noise_score(
    *,
    hardware_noise_model,
    fallback_noise_score: float,
    num_removed: int,
    candidate_cap: int,
    qiskit_shots: int,
) -> float:
    if hardware_noise_model is None:
        return float(fallback_noise_score)
    return float(
        hardware_noise_model.predict_min_noise_score(
            num_removed=int(num_removed),
            candidates_per_removed=int(candidate_cap),
            shots=int(qiskit_shots),
            families=("qaoa", "vqa_su2"),
            layers=(1, 2),
        )
    )


def should_evaluate_quantum(
    *,
    ctx: RepairContext,
    features: EntropyFeatures,
    ratio: float,
    rng: random.Random,
    qiskit_max_states: int,
) -> bool:
    if ctx.search_space > qiskit_max_states:
        return False
    if features.removed_count <= 0:
        return False
    if features.feasible_candidate_ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    return rng.random() <= max(0.0, float(ratio))


def row_from_evaluation(
    *,
    instance_tag: str,
    seed: int,
    iteration: int,
    context_idx: int,
    destroy_name: str,
    features: EntropyFeatures,
    ctx: RepairContext,
    evaluation: ActionEvaluation,
    quantum_backend: str,
    baseline_reward: float,
) -> Dict[str, object]:
    action = evaluation.action
    row: Dict[str, object] = {
        "instance_tag": instance_tag,
        "seed": seed,
        "iteration": iteration,
        "context_idx": context_idx,
        "destroy_name": destroy_name,
        "action_name": action.name,
        "repair_name": action.repair_name,
        "use_quantum": int(action.use_quantum),
        "family": action.family,
        "layers": action.layers,
        "shots": action.shots,
        "target_entropy": action.target_entropy,
        "quantum_backend": quantum_backend,
        "search_space": ctx.search_space,
        "reward": float(evaluation.reward) - float(baseline_reward),
        "raw_reward": evaluation.reward,
        "baseline_reward": baseline_reward,
        "exact_delta": evaluation.proposal.exact_delta,
        "approx_energy": evaluation.proposal.approx_energy,
        "feasible": int(evaluation.proposal.feasible),
        "runtime_sec": evaluation.runtime_sec,
        "context_noise_score": float(features.noise_score),
        "action_noise_score": float(evaluation.action_noise_score),
        "failed": int(evaluation.failed),
        "error": evaluation.error,
    }
    row.update(features_to_bandit_dict(features, search_space=ctx.search_space))
    return row


def save_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _next_context_idx(rows: Sequence[Mapping[str, object]]) -> int:
    if not rows:
        return 0
    values = []
    for row in rows:
        try:
            values.append(int(float(row.get("context_idx", -1))))
        except (TypeError, ValueError):
            continue
    return max(values, default=-1) + 1


def _context_baseline_reward(evaluations: Sequence[ActionEvaluation]) -> float:
    classical_rewards = [ev.reward for ev in evaluations if not ev.action.use_quantum and not ev.failed]
    if classical_rewards:
        return float(max(classical_rewards))
    valid_rewards = [ev.reward for ev in evaluations if not ev.failed]
    return float(max(valid_rewards)) if valid_rewards else 0.0


def collect_training_data(
    *,
    out_csv: Path,
    seeds: Iterable[int],
    sizes: Iterable[int],
    k_vehicles: int,
    iterations: int,
    remove_counts: Iterable[int],
    candidate_caps: Iterable[int],
    samples_per_action: int,
    tw_tightness: float,
    capacity_slack: float,
    noise_score: float,
    hardware_noise_model_path: Optional[Path],
    quantum_eval_ratio: float,
    use_qiskit_aer: bool,
    qiskit_shots: int,
    qiskit_max_states: int,
    exact_max_states: int,
    append: bool = False,
    tw_tightness_values: Optional[Iterable[float]] = None,
    capacity_slack_values: Optional[Iterable[float]] = None,
) -> List[Dict[str, object]]:
    extractor = EntropyFeatureExtractor(search_space_reference=10_000)
    destroyers = build_classical_destroy_pool()
    actions = default_actions(shots=qiskit_shots, qiskit_max_states=qiskit_max_states)
    quantum_portfolio = make_quantum_portfolio(
        use_qiskit_aer=use_qiskit_aer,
        shots=qiskit_shots,
        qiskit_max_states=qiskit_max_states,
    )
    hardware_noise_model = load_training_noise_model(hardware_noise_model_path)
    quantum_backend = "qiskit_aer" if use_qiskit_aer else "entropy_proxy"
    existing_rows: List[Dict[str, object]] = load_csv(out_csv) if append and out_csv.exists() else []
    rows: List[Dict[str, object]] = list(existing_rows)
    context_idx = _next_context_idx(existing_rows)

    tw_grid = list(tw_tightness_values) if tw_tightness_values is not None else [float(tw_tightness)]
    capacity_grid = list(capacity_slack_values) if capacity_slack_values is not None else [float(capacity_slack)]

    for tw_value in tw_grid:
        for capacity_value in capacity_grid:
            for remove_count in remove_counts:
                for candidate_cap in candidate_caps:
                    for n_requests in sizes:
                        for seed in seeds:
                            inst = generate_pdptw_instance(
                                n_requests,
                                k_vehicles,
                                seed=int(seed),
                                tw_tightness=float(tw_value),
                                capacity_slack=float(capacity_value),
                            )
                            routes = greedy_initial_solution(inst)
                            rng = random.Random(
                                100_000 * int(seed)
                                + int(n_requests)
                                + 101 * int(remove_count)
                                + 17 * int(candidate_cap)
                                + stable_int_hash(f"tw{tw_value:.4f}:cap{capacity_value:.4f}", modulo=99_991)
                            )
                            no_improve_iters = 0
                            best_objective = evaluate_solution(inst, routes).objective
                            instance_tag = (
                                f"R{n_requests}_K{k_vehicles}_seed{seed}_"
                                f"tw{float(tw_value):.2f}_cap{float(capacity_value):.2f}_"
                                f"rm{remove_count}_cc{candidate_cap}"
                            )

                            for iteration in range(iterations):
                                destroy_op = destroyers[iteration % len(destroyers)]
                                partial_routes, removed = destroy_op.destroy(inst, routes, int(remove_count), rng)
                                ctx = build_repair_context(inst, partial_routes, removed, int(candidate_cap))
                                stagnation = min(1.0, no_improve_iters / max(1, iterations))
                                context_noise_score = predict_context_noise_score(
                                    hardware_noise_model=hardware_noise_model,
                                    fallback_noise_score=noise_score,
                                    num_removed=len(removed),
                                    candidate_cap=int(candidate_cap),
                                    qiskit_shots=qiskit_shots,
                                )
                                features = extractor.extract(
                                    ctx,
                                    search_state={"stagnation": stagnation},
                                    noise_state={"noise_score": float(context_noise_score)},
                                )
                                quantum_allowed = should_evaluate_quantum(
                                    ctx=ctx,
                                    features=features,
                                    ratio=quantum_eval_ratio,
                                    rng=rng,
                                    qiskit_max_states=qiskit_max_states,
                                )

                                iteration_evaluations: List[ActionEvaluation] = []
                                for action in actions:
                                    if action.use_quantum and not quantum_allowed:
                                        continue
                                    if action.repair_name == "exact_micro_repair" and ctx.search_space > exact_max_states:
                                        continue

                                    action_noise_score = predict_action_noise_score(
                                        action=action,
                                        hardware_noise_model=hardware_noise_model,
                                        fallback_noise_score=context_noise_score,
                                        num_removed=len(removed),
                                        candidate_cap=int(candidate_cap),
                                        qiskit_shots=qiskit_shots,
                                    )
                                    action_rng = random.Random(
                                        stable_int_hash(f"{instance_tag}:{iteration}:{action.name}", modulo=2_147_483_647)
                                    )
                                    evaluation = evaluate_action(
                                        ctx=ctx,
                                        action=action,
                                        rng=action_rng,
                                        n_samples=samples_per_action,
                                        quantum_portfolio=quantum_portfolio,
                                        exact_max_states=exact_max_states,
                                        noise_score=action_noise_score,
                                    )
                                    iteration_evaluations.append(evaluation)

                                baseline_reward = _context_baseline_reward(iteration_evaluations)
                                for evaluation in iteration_evaluations:
                                    row = row_from_evaluation(
                                        instance_tag=instance_tag,
                                        seed=int(seed),
                                        iteration=iteration,
                                        context_idx=context_idx,
                                        destroy_name=destroy_op.name,
                                        features=features,
                                        ctx=ctx,
                                        evaluation=evaluation,
                                        quantum_backend=quantum_backend,
                                        baseline_reward=baseline_reward,
                                    )
                                    row["remove_count"] = int(remove_count)
                                    row["candidate_cap"] = int(candidate_cap)
                                    row["search_space_cap"] = int(candidate_cap) ** int(remove_count)
                                    row["tw_tightness"] = float(tw_value)
                                    row["capacity_slack"] = float(capacity_value)
                                    row["total_nodes"] = int(2 * n_requests + 1)
                                    row["noise_source"] = str(hardware_noise_model_path) if hardware_noise_model_path else "constant"
                                    rows.append(row)

                                best_classical_eval = min(
                                    (ev for ev in iteration_evaluations if not ev.action.use_quantum),
                                    key=lambda ev: ev.proposal.exact_delta,
                                    default=None,
                                )
                                if best_classical_eval is None:
                                    fallback_proposal = RegretKPairRepair(2).best_repair(ctx, rng, samples_per_action)
                                    routes = fallback_proposal.routes
                                else:
                                    routes = best_classical_eval.proposal.routes

                                deployed_eval = evaluate_solution(inst, routes)
                                if deployed_eval.objective < best_objective:
                                    best_objective = deployed_eval.objective
                                    no_improve_iters = 0
                                else:
                                    no_improve_iters += 1

                                context_idx += 1

    save_csv(out_csv, rows)
    return rows

def train_policy_model(
    *,
    data_csv: Path,
    model_path: Path,
    ridge_alpha: float,
    quantum_margin: float,
    qiskit_shots: int,
    qiskit_max_states: int,
    min_quantum_observations: Optional[int] = None,
) -> RidgeActionValueModel:
    """Train action-value coefficients.

    ``min_quantum_observations`` is accepted for CLI/API symmetry, but it is a
    deployment-time guard owned by ``HierarchicalBanditPolicy``. It does not
    change ridge coefficients; pass the same value again when loading/deploying
    the policy if you want a stricter quantum coverage threshold.
    """
    rows = load_csv(data_csv)
    actions = default_actions(shots=qiskit_shots, qiskit_max_states=qiskit_max_states)
    model = RidgeActionValueModel(
        feature_names=BANDIT_FEATURE_NAMES,
        action_specs=actions,
        ridge_alpha=ridge_alpha,
        quantum_margin=quantum_margin,
    )
    _ = min_quantum_observations
    model.fit(rows)
    model.save(model_path)
    return model


def load_bandit_policy(
    model_path: Path,
    *,
    exact_max_states: int = 128,
    hard_quantum_noise_cap: float = 0.85,
    min_quantum_observations: int = 5,
) -> HierarchicalBanditPolicy:
    model = RidgeActionValueModel.load(model_path)
    return HierarchicalBanditPolicy(
        model,
        exact_max_states=exact_max_states,
        hard_quantum_noise_cap=hard_quantum_noise_cap,
        min_quantum_observations=min_quantum_observations,
    )


def summarize_dataset(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"rows": 0}
    actions = sorted({str(row["action_name"]) for row in rows})
    quantum_rows = [row for row in rows if int(float(row.get("use_quantum", 0))) == 1]
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=float)
    raw_rewards = np.asarray([float(row.get("raw_reward", row["reward"])) for row in rows], dtype=float)
    return {
        "rows": len(rows),
        "actions": actions,
        "quantum_rows": len(quantum_rows),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "raw_reward_mean": float(np.mean(raw_rewards)),
        "raw_reward_std": float(np.std(raw_rewards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and train hierarchical contextual-bandit policies.")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Collect action-level repair training data.")
    collect.add_argument("--out-csv", type=Path, default=Path("runs/entropy_pdptw/training_actions.csv"))
    collect.add_argument("--seeds", type=str, default="1,2,3")
    collect.add_argument("--sizes", type=str, default="15,20")
    collect.add_argument("--k-vehicles", type=int, default=6)
    collect.add_argument("--iterations", type=int, default=100)
    collect.add_argument(
        "--remove-counts",
        type=str,
        default="2,3,4,5",
        help="Comma-separated removed-request counts. Final-training default: 2,3,4,5.",
    )
    collect.add_argument(
        "--candidate-caps",
        type=str,
        default="2,3,4",
        help="Comma-separated candidate caps. Final-training default: 2,3,4.",
    )
    collect.add_argument("--samples-per-action", type=int, default=32)
    collect.add_argument("--tw-tightness", type=float, default=0.5)
    collect.add_argument(
        "--tw-tightnesses",
        type=str,
        default=None,
        help="Optional comma-separated training grid, e.g. 0.25,0.50,0.75. Overrides --tw-tightness.",
    )
    collect.add_argument("--capacity-slack", type=float, default=0.30)
    collect.add_argument(
        "--capacity-slacks",
        type=str,
        default=None,
        help="Optional comma-separated training grid, e.g. 0.15,0.30,0.45. Overrides --capacity-slack.",
    )
    collect.add_argument("--noise-score", type=float, default=0.15)
    collect.add_argument(
        "--hardware-noise-model",
        type=Path,
        default=None,
        help="Optional JSON model from hardware_noise.py. If provided, context/action noise scores are predicted per circuit.",
    )
    collect.add_argument("--quantum-eval-ratio", type=float, default=0.10)
    collect.add_argument("--use-qiskit-aer", action="store_true")
    collect.add_argument("--qiskit-shots", type=int, default=256)
    collect.add_argument("--qiskit-max-states", type=int, default=1024)
    collect.add_argument("--exact-max-states", type=int, default=128)
    collect.add_argument("--append", action="store_true", help="Append rows to an existing CSV instead of overwriting it.")

    train = sub.add_parser("train", help="Train a ridge contextual-bandit reward model.")
    train.add_argument("--data-csv", type=Path, default=Path("runs/entropy_pdptw/training_actions.csv"))
    train.add_argument("--model-path", type=Path, default=Path("runs/entropy_pdptw/rl_policy.json"))
    train.add_argument("--ridge-alpha", type=float, default=1.0)
    train.add_argument("--quantum-margin", type=float, default=0.02)
    train.add_argument("--qiskit-shots", type=int, default=256)
    train.add_argument("--qiskit-max-states", type=int, default=1024)
    train.add_argument(
        "--min-quantum-observations",
        type=int,
        default=None,
        help="Accepted for API symmetry only. Quantum data coverage is enforced when loading/deploying the policy.",
    )

    inspect = sub.add_parser("inspect", help="Summarize a training CSV.")
    inspect.add_argument("--data-csv", type=Path, default=Path("runs/entropy_pdptw/training_actions.csv"))

    args = parser.parse_args()

    if args.command == "collect":
        rows = collect_training_data(
            out_csv=args.out_csv,
            seeds=parse_int_list(args.seeds),
            sizes=parse_int_list(args.sizes),
            k_vehicles=args.k_vehicles,
            iterations=args.iterations,
            remove_counts=parse_int_list(args.remove_counts),
            candidate_caps=parse_int_list(args.candidate_caps),
            samples_per_action=args.samples_per_action,
            tw_tightness=args.tw_tightness,
            capacity_slack=args.capacity_slack,
            noise_score=args.noise_score,
            hardware_noise_model_path=args.hardware_noise_model,
            quantum_eval_ratio=args.quantum_eval_ratio,
            use_qiskit_aer=args.use_qiskit_aer,
            qiskit_shots=args.qiskit_shots,
            qiskit_max_states=args.qiskit_max_states,
            exact_max_states=args.exact_max_states,
            append=args.append,
            tw_tightness_values=parse_float_list(args.tw_tightnesses) if args.tw_tightnesses else None,
            capacity_slack_values=parse_float_list(args.capacity_slacks) if args.capacity_slacks else None,
        )
        print(json.dumps(summarize_dataset(rows), indent=2))
        print(f"Wrote {len(rows)} rows to {args.out_csv}")

    elif args.command == "train":
        model = train_policy_model(
            data_csv=args.data_csv,
            model_path=args.model_path,
            ridge_alpha=args.ridge_alpha,
            quantum_margin=args.quantum_margin,
            qiskit_shots=args.qiskit_shots,
            qiskit_max_states=args.qiskit_max_states,
            min_quantum_observations=args.min_quantum_observations,
        )
        print(json.dumps({"model_path": str(args.model_path), "action_counts": model.action_counts}, indent=2))

    elif args.command == "inspect":
        rows = load_csv(args.data_csv)
        print(json.dumps(summarize_dataset(rows), indent=2))


if __name__ == "__main__":
    main()
