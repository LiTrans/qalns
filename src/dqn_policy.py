
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from entropy import EntropyFeatures
from policy import CircuitConfig, RepairDecision
from rl_training import RepairAction, default_actions, features_to_bandit_dict


def feature_names() -> List[str]:
    return [
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
        "iteration_fraction",
        "accepted_last_step",
        "improved_last_step",
        "temperature_fraction",
    ]


def encode_state(
    features: EntropyFeatures,
    *,
    search_space: int,
    iteration_fraction: float = 0.0,
    accepted_last_step: float = 0.0,
    improved_last_step: float = 0.0,
    temperature_fraction: float = 1.0,
) -> np.ndarray:
    bandit = features_to_bandit_dict(features, search_space=search_space)
    bandit["iteration_fraction"] = float(iteration_fraction)
    bandit["accepted_last_step"] = float(accepted_last_step)
    bandit["improved_last_step"] = float(improved_last_step)
    bandit["temperature_fraction"] = float(temperature_fraction)
    return np.asarray([float(bandit.get(name, 0.0)) for name in feature_names()], dtype=float)


def mask_invalid_q_values(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(q_values, dtype=float).copy()
    invalid = np.asarray(mask, dtype=float) <= 0.0
    masked[invalid] = -1e18
    return masked


class MLPQNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(int(seed))
        self.w1 = rng.normal(0.0, 0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = rng.normal(0.0, 0.1, size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim, dtype=float)

    def copy(self) -> "MLPQNetwork":
        clone = MLPQNetwork(self.w1.shape[0], self.w1.shape[1], self.w2.shape[1], seed=0)
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        return clone

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def train_batch(self, x: np.ndarray, target_q: np.ndarray, *, learning_rate: float) -> float:
        x = np.asarray(x, dtype=float)
        target_q = np.asarray(target_q, dtype=float)

        hidden_pre = x @ self.w1 + self.b1
        hidden = np.maximum(0.0, hidden_pre)
        pred = hidden @ self.w2 + self.b2

        error = pred - target_q
        loss = float(np.mean(np.square(error)))

        batch_scale = 2.0 / max(1, x.shape[0])
        d_pred = batch_scale * error
        d_w2 = hidden.T @ d_pred
        d_b2 = np.sum(d_pred, axis=0)

        d_hidden = d_pred @ self.w2.T
        d_hidden[hidden_pre <= 0.0] = 0.0
        d_w1 = x.T @ d_hidden
        d_b1 = np.sum(d_hidden, axis=0)

        self.w1 -= float(learning_rate) * d_w1
        self.b1 -= float(learning_rate) * d_b1
        self.w2 -= float(learning_rate) * d_w2
        self.b2 -= float(learning_rate) * d_b2
        return loss

    def to_json_dict(self) -> dict:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "MLPQNetwork":
        w1 = np.asarray(payload["w1"], dtype=float)
        b1 = np.asarray(payload["b1"], dtype=float)
        w2 = np.asarray(payload["w2"], dtype=float)
        b2 = np.asarray(payload["b2"], dtype=float)
        model = cls(w1.shape[0], w1.shape[1], w2.shape[1], seed=0)
        model.w1 = w1
        model.b1 = b1
        model.w2 = w2
        model.b2 = b2
        return model


@dataclass
class DQNModelMetadata:
    state_feature_names: List[str]
    action_names: List[str]
    action_specs: List[dict]
    state_mean: List[float]
    state_std: List[float]
    hidden_dim: int
    gamma: float
    exact_max_states: int
    hard_quantum_noise_cap: float
    min_quantum_observations: int


class DoubleDQNPolicy:
    def __init__(
        self,
        *,
        q_network: MLPQNetwork,
        action_specs: Sequence[RepairAction],
        state_mean: Sequence[float],
        state_std: Sequence[float],
        gamma: float = 0.99,
        exact_max_states: int = 128,
        hard_quantum_noise_cap: float = 0.85,
        min_quantum_observations: int = 0,
    ) -> None:
        self.q_network = q_network
        self.action_specs = list(action_specs)
        self.state_mean = np.asarray(list(state_mean), dtype=float)
        self.state_std = np.asarray(list(state_std), dtype=float)
        self.gamma = float(gamma)
        self.exact_max_states = int(exact_max_states)
        self.hard_quantum_noise_cap = float(hard_quantum_noise_cap)
        self.min_quantum_observations = int(min_quantum_observations)
        self.action_counts: Dict[str, int] = {action.name: max(1, self.min_quantum_observations) for action in self.action_specs}

    def _standardize_state(self, state: np.ndarray) -> np.ndarray:
        std = np.where(np.abs(self.state_std) <= 1e-12, 1.0, self.state_std)
        return (np.asarray(state, dtype=float) - self.state_mean) / std

    def available_action_mask(self, features: EntropyFeatures, *, search_space: int) -> np.ndarray:
        mask = np.ones(len(self.action_specs), dtype=float)
        for idx, action in enumerate(self.action_specs):
            if action.repair_name == "exact_micro_repair" and int(search_space) > self.exact_max_states:
                mask[idx] = 0.0
                continue
            if action.use_quantum:
                if int(search_space) > int(action.max_states):
                    mask[idx] = 0.0
                    continue
                if float(features.noise_score) > self.hard_quantum_noise_cap:
                    mask[idx] = 0.0
                    continue
                if self.action_counts.get(action.name, 0) < self.min_quantum_observations:
                    mask[idx] = 0.0
                    continue
        return mask

    def choose_action_index(
        self,
        state: np.ndarray,
        *,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
        rng: random.Random | None = None,
    ) -> int:
        rng = rng or random
        valid_indices = [idx for idx, value in enumerate(np.asarray(action_mask, dtype=float)) if value > 0.0]
        if not valid_indices:
            return 1 if len(self.action_specs) > 1 else 0
        if rng.random() < float(epsilon):
            return int(rng.choice(valid_indices))
        q_values = self.q_network.predict(self._standardize_state(np.asarray(state, dtype=float))[None, :])[0]
        masked = mask_invalid_q_values(q_values, np.asarray(action_mask, dtype=float))
        return int(np.argmax(masked))

    def choose(
        self,
        features: EntropyFeatures,
        *,
        search_space: int,
        iteration_fraction: float = 0.0,
        accepted_last_step: float = 0.0,
        improved_last_step: float = 0.0,
        temperature_fraction: float = 1.0,
        epsilon: float = 0.0,
        rng: random.Random | None = None,
    ) -> RepairDecision:
        state = encode_state(
            features,
            search_space=search_space,
            iteration_fraction=iteration_fraction,
            accepted_last_step=accepted_last_step,
            improved_last_step=improved_last_step,
            temperature_fraction=temperature_fraction,
        )
        mask = self.available_action_mask(features, search_space=search_space)
        action_idx = self.choose_action_index(state, action_mask=mask, epsilon=epsilon, rng=rng)
        return self.action_specs[action_idx].to_decision()

    def to_json_dict(self) -> dict:
        metadata = DQNModelMetadata(
            state_feature_names=feature_names(),
            action_names=[action.name for action in self.action_specs],
            action_specs=[asdict(action) for action in self.action_specs],
            state_mean=self.state_mean.tolist(),
            state_std=self.state_std.tolist(),
            hidden_dim=int(self.q_network.w1.shape[1]),
            gamma=self.gamma,
            exact_max_states=self.exact_max_states,
            hard_quantum_noise_cap=self.hard_quantum_noise_cap,
            min_quantum_observations=self.min_quantum_observations,
        )
        return {
            "metadata": asdict(metadata),
            "q_network": self.q_network.to_json_dict(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DoubleDQNPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        return cls(
            q_network=MLPQNetwork.from_json_dict(payload["q_network"]),
            action_specs=[RepairAction(**item) for item in metadata["action_specs"]],
            state_mean=metadata["state_mean"],
            state_std=metadata["state_std"],
            gamma=float(metadata.get("gamma", 0.99)),
            exact_max_states=int(metadata.get("exact_max_states", 128)),
            hard_quantum_noise_cap=float(metadata.get("hard_quantum_noise_cap", 0.85)),
            min_quantum_observations=int(metadata.get("min_quantum_observations", 0)),
        )


def default_dqn_actions(*, shots: int = 256, qiskit_max_states: int = 256, include_exact: bool = True) -> List[RepairAction]:
    return default_actions(shots=shots, qiskit_max_states=qiskit_max_states, include_exact=include_exact)


def load_dqn_policy(
    model_path: Path,
    *,
    exact_max_states: int = 128,
    hard_quantum_noise_cap: float = 0.85,
    min_quantum_observations: int = 0,
) -> DoubleDQNPolicy:
    policy = DoubleDQNPolicy.load(model_path)
    policy.exact_max_states = int(exact_max_states)
    policy.hard_quantum_noise_cap = float(hard_quantum_noise_cap)
    policy.min_quantum_observations = int(min_quantum_observations)
    return policy
