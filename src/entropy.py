from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from pdptw import route_load_slacks
from repair import RepairContext


@dataclass(frozen=True)
class EntropyFeatures:
    insertion_cost_entropy: float
    conflict_entropy: float
    slack_entropy: float
    load_slack_entropy: float
    structural_entropy: float
    feasible_candidate_ratio: float
    conflict_density: float
    search_space_log: float
    removed_count: int
    stagnation: float
    noise_score: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "insertion_cost_entropy": self.insertion_cost_entropy,
            "conflict_entropy": self.conflict_entropy,
            "slack_entropy": self.slack_entropy,
            "load_slack_entropy": self.load_slack_entropy,
            "structural_entropy": self.structural_entropy,
            "feasible_candidate_ratio": self.feasible_candidate_ratio,
            "conflict_density": self.conflict_density,
            "search_space_log": self.search_space_log,
            "removed_count": float(self.removed_count),
            "stagnation": self.stagnation,
            "noise_score": self.noise_score,
        }


def normalized_shannon(values: Sequence[float], *, bins: int = 8) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0

    if float(np.max(arr) - np.min(arr)) <= 1e-12:
        return 0.0

    support = max(2, min(int(bins), int(arr.size)))
    counts, _ = np.histogram(arr, bins=support)
    probs = counts[counts > 0].astype(float)
    probs /= float(np.sum(probs))
    entropy = -float(np.sum(probs * np.log(probs + 1e-15)))
    return float(min(1.0, max(0.0, entropy / math.log(support))))


def binary_entropy(probability: float) -> float:
    p = min(1.0, max(0.0, float(probability)))
    if p <= 1e-12 or p >= 1.0 - 1e-12:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def normalized_probability_entropy(probabilities: Sequence[float], *, support_tol: float = 1e-6) -> float:
    probs = np.asarray(list(probabilities), dtype=float)
    probs = probs[np.isfinite(probs)]
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0

    probs /= float(np.sum(probs))
    entropy = -float(np.sum(probs * np.log(probs + 1e-15)))
    effective_support = max(2, int(np.sum(probs > float(support_tol))))
    return float(min(1.0, max(0.0, entropy / math.log(effective_support))))


def softmax_entropy(costs: Iterable[float], *, temperature: float = 1.0, support_tol: float = 1e-6) -> float:
    arr = np.asarray(list(costs), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    if float(np.max(arr) - np.min(arr)) <= 1e-12:
        return 1.0

    logits = -(arr - float(np.min(arr))) / max(1e-9, float(temperature))
    logits -= float(np.max(logits))
    probs = np.exp(logits)
    probs /= float(np.sum(probs))
    return normalized_probability_entropy(probs, support_tol=support_tol)


class EntropyFeatureExtractor:
    def __init__(self, *, search_space_reference: int = 10_000) -> None:
        self.search_space_reference = max(2, int(search_space_reference))

    def extract(
        self,
        ctx: RepairContext,
        *,
        search_state: Dict[str, float] | None = None,
        noise_state: Dict[str, float] | None = None,
    ) -> EntropyFeatures:
        search_state = search_state or {}
        noise_state = noise_state or {}

        insertion_entropies = [
            softmax_entropy([candidate.delta_local for candidate in candidates])
            for candidates in ctx.candidate_sets.values()
        ]
        insertion_cost_entropy = float(np.mean(insertion_entropies)) if insertion_entropies else 0.0

        feasible_count = sum(candidate.local_feasible for candidates in ctx.candidate_sets.values() for candidate in candidates)
        total_count = sum(len(candidates) for candidates in ctx.candidate_sets.values())
        feasible_candidate_ratio = feasible_count / max(1, total_count)

        slack_values = []
        for request_id in ctx.removed_requests:
            pickup_id, delivery_id = ctx.instance.request_nodes(request_id)
            pickup = ctx.instance.nodes[pickup_id]
            delivery = ctx.instance.nodes[delivery_id]
            slack_values.extend([pickup.tw_close - pickup.tw_open, delivery.tw_close - delivery.tw_open])

        conflict_entropy = binary_entropy(ctx.conflict_density)
        slack_entropy = normalized_shannon(slack_values)
        load_slack_entropy = normalized_shannon(route_load_slacks(ctx.instance, ctx.partial_routes))

        structural_entropy = float(
            np.mean(
                [
                    insertion_cost_entropy,
                    conflict_entropy,
                    slack_entropy,
                    load_slack_entropy,
                ]
            )
        )
        search_space_log = math.log1p(ctx.search_space) / math.log1p(self.search_space_reference)
        search_space_log = min(1.0, max(0.0, search_space_log))

        return EntropyFeatures(
            insertion_cost_entropy=insertion_cost_entropy,
            conflict_entropy=conflict_entropy,
            slack_entropy=slack_entropy,
            load_slack_entropy=load_slack_entropy,
            structural_entropy=structural_entropy,
            feasible_candidate_ratio=feasible_candidate_ratio,
            conflict_density=float(ctx.conflict_density),
            search_space_log=search_space_log,
            removed_count=len(ctx.removed_requests),
            stagnation=float(search_state.get("stagnation", 0.0)),
            noise_score=float(noise_state.get("noise_score", 0.0)),
        )
