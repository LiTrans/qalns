from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from entropy import EntropyFeatureExtractor, EntropyFeatures
from repair import (
    EntropyBiasedSamplerRepair,
    ExactMicroSearchRepair,
    GreedyPairRepair,
    RegretKPairRepair,
    RepairContext,
    RepairOperator,
    RepairProposal,
)


@dataclass(frozen=True)
class CircuitConfig:
    family: str
    layers: int
    shots: int
    target_entropy: float


@dataclass(frozen=True)
class RepairDecision:
    repair_name: str
    use_quantum: bool
    circuit: Optional[CircuitConfig]
    reason: str


class CircuitSelector:
    def __init__(
        self,
        *,
        low_noise_threshold: float = 0.25,
        high_entropy_threshold: float = 0.60,
        shots: int = 256,
    ) -> None:
        self.low_noise_threshold = float(low_noise_threshold)
        self.high_entropy_threshold = float(high_entropy_threshold)
        self.shots = int(shots)

    def choose(self, features: EntropyFeatures) -> CircuitConfig:
        if features.noise_score > self.low_noise_threshold:
            return CircuitConfig("qaoa", layers=1, shots=self.shots, target_entropy=0.45)

        if features.structural_entropy >= self.high_entropy_threshold:
            return CircuitConfig("vqa_su2", layers=2, shots=self.shots, target_entropy=0.65)

        return CircuitConfig("qaoa", layers=1, shots=self.shots, target_entropy=0.50)


class QuantumRepairPortfolio(RepairOperator):
    name = "quantum_repair"

    def __init__(
        self,
        operators_by_family: Dict[str, RepairOperator],
        *,
        fallback: Optional[RepairOperator] = None,
    ) -> None:
        self.operators_by_family = dict(operators_by_family)
        self.fallback = fallback or EntropyBiasedSamplerRepair(max_states=512, beta=1.0)

    def select(self, circuit: Optional[CircuitConfig]) -> RepairOperator:
        if circuit is None:
            return self.fallback

        keys = [
            f"{circuit.family}:L{circuit.layers}:S{circuit.shots}",
            f"{circuit.family}:L{circuit.layers}",
            circuit.family,
        ]
        for key in keys:
            if key in self.operators_by_family:
                operator = self.operators_by_family[key]
                if hasattr(operator, "set_target_entropy"):
                    operator.set_target_entropy(circuit.target_entropy)
                else:
                    setattr(operator, "target_entropy", circuit.target_entropy)
                return operator

        return self.fallback

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        return self.fallback.sample(ctx, rng, n_samples)


class EntropyRepairPolicy:
    def __init__(
        self,
        *,
        exact_max_states: int = 128,
        quantum_min_entropy: float = 0.35,
        quantum_max_entropy: float = 0.85,
        quantum_min_conflict_density: float = 0.05,
        quantum_max_noise: float = 0.45,
        quantum_max_removed: int = 4,
        circuit_selector: Optional[CircuitSelector] = None,
    ) -> None:
        self.exact_max_states = int(exact_max_states)
        self.quantum_min_entropy = float(quantum_min_entropy)
        self.quantum_max_entropy = float(quantum_max_entropy)
        self.quantum_min_conflict_density = float(quantum_min_conflict_density)
        self.quantum_max_noise = float(quantum_max_noise)
        self.quantum_max_removed = int(quantum_max_removed)
        self.circuit_selector = circuit_selector or CircuitSelector()

    def choose(self, features: EntropyFeatures, *, search_space: int) -> RepairDecision:
        if search_space <= self.exact_max_states:
            return RepairDecision("exact_micro_repair", False, None, "tiny search space")

        entropy_ok = self.quantum_min_entropy <= features.structural_entropy <= self.quantum_max_entropy
        conflict_ok = features.conflict_density >= self.quantum_min_conflict_density
        noise_ok = features.noise_score <= self.quantum_max_noise
        size_ok = features.removed_count <= self.quantum_max_removed

        if entropy_ok and conflict_ok and noise_ok and size_ok:
            circuit = self.circuit_selector.choose(features)
            return RepairDecision("quantum_repair", True, circuit, "entropy/noise conditions met")

        if features.insertion_cost_entropy < 0.25 and features.feasible_candidate_ratio > 0.60:
            return RepairDecision("greedy_repair", False, None, "low insertion ambiguity")

        return RepairDecision("regret2_repair", False, None, "classical fallback")


class EntropyAwareRepairOperator(RepairOperator):
    name = "entropy_aware_repair"

    def __init__(
        self,
        *,
        extractor: Optional[EntropyFeatureExtractor] = None,
        policy: Optional[EntropyRepairPolicy] = None,
        operator_registry: Optional[Dict[str, RepairOperator]] = None,
        quantum_operator: Optional[RepairOperator] = None,
        search_state: Optional[Dict[str, float]] = None,
        noise_state: Optional[Dict[str, float]] = None,
        quantum_eval_ratio: float = 1.0,
    ) -> None:
        self.extractor = extractor or EntropyFeatureExtractor()
        self.policy = policy or EntropyRepairPolicy()
        self.search_state = search_state if search_state is not None else {}
        self.noise_state = noise_state if noise_state is not None else {}
        self.quantum_eval_ratio = max(0.0, min(1.0, float(quantum_eval_ratio)))
        self.last_decision: Optional[RepairDecision] = None
        self.last_features: Optional[EntropyFeatures] = None

        self.registry: Dict[str, RepairOperator] = {
            "greedy_repair": GreedyPairRepair(),
            "regret2_repair": RegretKPairRepair(2),
            "regret3_repair": RegretKPairRepair(3),
            "exact_micro_repair": ExactMicroSearchRepair(max_states=self.policy.exact_max_states),
            "quantum_repair": quantum_operator or EntropyBiasedSamplerRepair(max_states=512, beta=1.0),
        }
        if operator_registry:
            self.registry.update(operator_registry)

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        features = self.extractor.extract(ctx, search_state=self.search_state, noise_state=self.noise_state)
        try:
            decision = self.policy.choose(
                features,
                search_space=ctx.search_space,
                iteration_fraction=float(self.search_state.get("iteration_fraction", 0.0)),
                accepted_last_step=float(self.search_state.get("accepted_last_step", 0.0)),
                improved_last_step=float(self.search_state.get("improved_last_step", 0.0)),
                temperature_fraction=float(self.search_state.get("temperature_fraction", 1.0)),
            )
        except TypeError:
            decision = self.policy.choose(features, search_space=ctx.search_space)
        if decision.use_quantum and rng.random() > self.quantum_eval_ratio:
            decision = RepairDecision(
                "regret2_repair",
                False,
                None,
                f"quantum skipped by quantum_eval_ratio={self.quantum_eval_ratio:.3f}",
            )
        self.last_features = features
        self.last_decision = decision

        operator = self.registry.get(decision.repair_name)
        if isinstance(operator, QuantumRepairPortfolio):
            operator = operator.select(decision.circuit)
        if operator is None:
            operator = self.registry["regret2_repair"]

        return operator.sample(ctx, rng, n_samples)
