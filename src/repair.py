from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from qiskit.circuit.library import QAOAAnsatz, efficient_su2
    from qiskit.quantum_info import SparsePauliOp
except Exception:  # pragma: no cover - optional dependency
    QAOAAnsatz = None
    efficient_su2 = None
    SparsePauliOp = None

try:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as _qiskit_generate_preset_pass_manager
except Exception:  # pragma: no cover - optional dependency
    _qiskit_generate_preset_pass_manager = None

from pdptw import (
    PDPTWInstance,
    SolutionEval,
    deep_copy_routes,
    evaluate_route,
    evaluate_solution,
    euclidean,
    insert_request_into_route,
)


@dataclass(frozen=True)
class PairCandidate:
    request: int
    vehicle: int
    pickup_slot: int
    delivery_slot: int
    delta_local: float
    local_feasible: bool


@dataclass
class RepairContext:
    instance: PDPTWInstance
    partial_routes: List[List[int]]
    removed_requests: List[int]
    base_eval: SolutionEval
    candidate_sets: Dict[int, List[PairCandidate]]
    pair_gamma: Dict[Tuple[PairCandidate, PairCandidate], float]
    conflict_density: float
    search_space: int


@dataclass
class RepairProposal:
    combo: Tuple[PairCandidate, ...]
    exact_delta: float
    approx_energy: float
    feasible: bool
    routes: List[List[int]]


class DestroyOperator:
    name = "destroy_base"

    def destroy(
        self,
        inst: PDPTWInstance,
        routes: List[List[int]],
        remove_count: int,
        rng: random.Random,
    ) -> Tuple[List[List[int]], List[int]]:
        raise NotImplementedError


class RandomRequestRemoval(DestroyOperator):
    name = "random_removal"

    def destroy(
        self,
        inst: PDPTWInstance,
        routes: List[List[int]],
        remove_count: int,
        rng: random.Random,
    ) -> Tuple[List[List[int]], List[int]]:
        present = sorted({inst.nodes[node_id].request_id for route in routes for node_id in route})
        removed = rng.sample(present, min(remove_count, len(present))) if present else []
        removed_set = set(removed)
        partial = [[node_id for node_id in route if inst.nodes[node_id].request_id not in removed_set] for route in routes]
        return partial, sorted(removed)


class ShawRequestRemoval(DestroyOperator):
    name = "shaw_removal"

    def relatedness(self, inst: PDPTWInstance, req_a: int, req_b: int) -> float:
        pickup_a, delivery_a = inst.request_nodes(req_a)
        pickup_b, delivery_b = inst.request_nodes(req_b)
        node_pa, node_da = inst.nodes[pickup_a], inst.nodes[delivery_a]
        node_pb, node_db = inst.nodes[pickup_b], inst.nodes[delivery_b]
        geo = euclidean((node_pa.x, node_pa.y), (node_pb.x, node_pb.y))
        geo += euclidean((node_da.x, node_da.y), (node_db.x, node_db.y))
        tw = abs(node_pa.tw_open - node_pb.tw_open) + abs(node_da.tw_open - node_db.tw_open)
        qty = abs(inst.requests[req_a].quantity - inst.requests[req_b].quantity)
        return geo + 0.25 * tw + 5.0 * qty

    def destroy(
        self,
        inst: PDPTWInstance,
        routes: List[List[int]],
        remove_count: int,
        rng: random.Random,
    ) -> Tuple[List[List[int]], List[int]]:
        present = sorted({inst.nodes[node_id].request_id for route in routes for node_id in route})
        if not present or remove_count <= 0:
            return deep_copy_routes(routes), []

        seed = rng.choice(present)
        related = sorted((req_id for req_id in present if req_id != seed), key=lambda req_id: self.relatedness(inst, seed, req_id))
        removed = [seed] + related[: max(0, min(remove_count, len(present)) - 1)]
        removed_set = set(removed)
        partial = [[node_id for node_id in route if inst.nodes[node_id].request_id not in removed_set] for route in routes]
        return partial, sorted(removed)


class WorstRequestRemoval(DestroyOperator):
    name = "worst_removal"

    def destroy(
        self,
        inst: PDPTWInstance,
        routes: List[List[int]],
        remove_count: int,
        rng: random.Random,
    ) -> Tuple[List[List[int]], List[int]]:
        present = sorted({inst.nodes[node_id].request_id for route in routes for node_id in route})
        if not present or remove_count <= 0:
            return deep_copy_routes(routes), []

        current_objective = evaluate_solution(inst, routes).objective
        scores: List[Tuple[float, int]] = []
        for request_id in present:
            partial = [[node_id for node_id in route if inst.nodes[node_id].request_id != request_id] for route in routes]
            delta = current_objective - evaluate_solution(inst, partial, allow_partial=True).objective
            scores.append((delta + rng.random() * 1e-3, request_id))

        scores.sort(reverse=True)
        removed = [request_id for _, request_id in scores[: min(remove_count, len(scores))]]
        removed_set = set(removed)
        partial = [[node_id for node_id in route if inst.nodes[node_id].request_id not in removed_set] for route in routes]
        return partial, sorted(removed)


def apply_candidates_to_routes(
    partial_routes: List[List[int]],
    combo: Sequence[PairCandidate],
    inst: PDPTWInstance,
) -> List[List[int]]:
    routes = deep_copy_routes(partial_routes)
    by_route: Dict[int, List[PairCandidate]] = defaultdict(list)
    for candidate in combo:
        by_route[candidate.vehicle].append(candidate)

    for route_idx, candidates in by_route.items():
        base_route = partial_routes[route_idx]
        events: Dict[int, List[Tuple[int, bool, int]]] = defaultdict(list)
        for candidate in candidates:
            pickup_id, delivery_id = inst.request_nodes(candidate.request)
            events[candidate.pickup_slot].append((candidate.request, False, pickup_id))
            events[candidate.delivery_slot].append((candidate.request, True, delivery_id))

        rebuilt: List[int] = []
        for slot in range(len(base_route) + 1):
            rebuilt.extend(node_id for _, _, node_id in sorted(events.get(slot, []), key=lambda item: (item[1], item[0], item[2])))
            if slot < len(base_route):
                rebuilt.append(base_route[slot])
        routes[route_idx] = rebuilt

    return routes


def generate_pair_candidates(
    inst: PDPTWInstance,
    partial_routes: List[List[int]],
    removed_requests: List[int],
    candidate_cap: int,
) -> Dict[int, List[PairCandidate]]:
    base_eval = evaluate_solution(inst, partial_routes, allow_partial=True)
    candidate_sets: Dict[int, List[PairCandidate]] = {}

    for request_id in removed_requests:
        candidates: List[PairCandidate] = []
        for route_idx, route in enumerate(partial_routes):
            base_objective = base_eval.route_evals[route_idx].objective
            for pickup_slot in range(len(route) + 1):
                for delivery_slot in range(pickup_slot, len(route) + 1):
                    trial_route = insert_request_into_route(inst, route, request_id, pickup_slot, delivery_slot)
                    route_eval = evaluate_route(inst, trial_route, route_idx)
                    candidates.append(
                        PairCandidate(
                            request=request_id,
                            vehicle=route_idx,
                            pickup_slot=pickup_slot,
                            delivery_slot=delivery_slot,
                            delta_local=route_eval.objective - base_objective,
                            local_feasible=route_eval.feasible,
                        )
                    )

        candidates.sort(key=lambda item: (item.delta_local + (0.0 if item.local_feasible else 1e6), item.vehicle, item.pickup_slot, item.delivery_slot))
        candidate_sets[request_id] = candidates[: max(1, candidate_cap)]

    return candidate_sets


def compute_pair_gamma(
    inst: PDPTWInstance,
    partial_routes: List[List[int]],
    candidate_sets: Dict[int, List[PairCandidate]],
    *,
    infeasible_pair_penalty: float = 500.0,
) -> Tuple[Dict[Tuple[PairCandidate, PairCandidate], float], float]:
    gammas: Dict[Tuple[PairCandidate, PairCandidate], float] = {}
    total_pairs = 0
    conflicting_pairs = 0
    request_ids = sorted(candidate_sets)

    for idx_a, request_a in enumerate(request_ids):
        for request_b in request_ids[idx_a + 1 :]:
            for candidate_a in candidate_sets[request_a]:
                for candidate_b in candidate_sets[request_b]:
                    total_pairs += 1
                    if candidate_a.vehicle != candidate_b.vehicle:
                        gamma = 0.0
                    else:
                        base_route = partial_routes[candidate_a.vehicle]
                        base_objective = evaluate_route(inst, base_route, candidate_a.vehicle).objective
                        trial_routes = apply_candidates_to_routes(partial_routes, (candidate_a, candidate_b), inst)
                        pair_eval = evaluate_route(inst, trial_routes[candidate_a.vehicle], candidate_a.vehicle)
                        gamma = pair_eval.objective - base_objective - candidate_a.delta_local - candidate_b.delta_local
                        if not pair_eval.feasible:
                            gamma += infeasible_pair_penalty
                            conflicting_pairs += 1
                        elif abs(gamma) > 1e-6:
                            conflicting_pairs += 1

                    gammas[(candidate_a, candidate_b)] = gamma
                    gammas[(candidate_b, candidate_a)] = gamma

    return gammas, conflicting_pairs / max(1, total_pairs)


def build_repair_context(
    inst: PDPTWInstance,
    partial_routes: List[List[int]],
    removed_requests: List[int],
    candidate_cap: int,
) -> RepairContext:
    base_eval = evaluate_solution(inst, partial_routes, allow_partial=True)
    candidate_sets = generate_pair_candidates(inst, partial_routes, removed_requests, candidate_cap)
    pair_gamma, conflict_density = compute_pair_gamma(inst, partial_routes, candidate_sets)
    search_space = 1
    for request_id in removed_requests:
        search_space *= max(1, len(candidate_sets[request_id]))

    return RepairContext(
        instance=inst,
        partial_routes=partial_routes,
        removed_requests=removed_requests,
        base_eval=base_eval,
        candidate_sets=candidate_sets,
        pair_gamma=pair_gamma,
        conflict_density=conflict_density,
        search_space=search_space,
    )


def combo_approx_energy(ctx: RepairContext, combo: Sequence[PairCandidate]) -> float:
    energy = sum(candidate.delta_local for candidate in combo)
    for candidate_a, candidate_b in itertools.combinations(combo, 2):
        energy += ctx.pair_gamma.get((candidate_a, candidate_b), 0.0)
    return energy


def evaluate_combo(ctx: RepairContext, combo: Sequence[PairCandidate]) -> RepairProposal:
    combo_tuple = tuple(combo)
    routes = apply_candidates_to_routes(ctx.partial_routes, combo_tuple, ctx.instance)
    solution_eval = evaluate_solution(ctx.instance, routes)
    return RepairProposal(
        combo=combo_tuple,
        exact_delta=solution_eval.objective - ctx.base_eval.objective,
        approx_energy=combo_approx_energy(ctx, combo_tuple),
        feasible=solution_eval.feasible,
        routes=routes,
    )


def evaluate_partial_combo_objective(ctx: RepairContext, combo: Sequence[PairCandidate]) -> float:
    routes = apply_candidates_to_routes(ctx.partial_routes, tuple(combo), ctx.instance)
    return evaluate_solution(ctx.instance, routes, allow_partial=True).objective


def enumerate_all_combos(ctx: RepairContext, max_states: int = 50_000) -> List[Tuple[PairCandidate, ...]]:
    pools = [ctx.candidate_sets[request_id] for request_id in ctx.removed_requests]
    search_space = math.prod(len(pool) for pool in pools) if pools else 1
    if search_space > max_states:
        raise ValueError(f"Search space too large: {search_space} > {max_states}")
    return list(itertools.product(*pools))


def truth_table(ctx: RepairContext, max_states: int = 50_000) -> List[RepairProposal]:
    proposals = [evaluate_combo(ctx, combo) for combo in enumerate_all_combos(ctx, max_states=max_states)]
    proposals.sort(key=lambda proposal: (not proposal.feasible, proposal.exact_delta, proposal.approx_energy))
    return proposals


class RepairOperator:
    name = "repair_base"

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        raise NotImplementedError

    def best_repair(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> RepairProposal:
        proposals = self.sample(ctx, rng, max(1, n_samples))
        feasible = [proposal for proposal in proposals if proposal.feasible]
        return min(feasible or proposals, key=lambda proposal: proposal.exact_delta)


class RandomComboRepair(RepairOperator):
    name = "random_repair"

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        pools = [ctx.candidate_sets[request_id] for request_id in ctx.removed_requests]
        return [evaluate_combo(ctx, tuple(rng.choice(pool) for pool in pools)) for _ in range(max(1, n_samples))]


class GreedyPairRepair(RepairOperator):
    name = "greedy_repair"

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        chosen: List[PairCandidate] = []

        for request_id in ctx.removed_requests:
            best_candidate: Optional[PairCandidate] = None
            best_objective = float("inf")

            for candidate in ctx.candidate_sets[request_id]:
                objective = evaluate_partial_combo_objective(ctx, [*chosen, candidate])
                if objective < best_objective:
                    best_objective = objective
                    best_candidate = candidate

            if best_candidate is None:
                raise RuntimeError(f"No insertion candidate found for request {request_id}")

            chosen.append(best_candidate)

        proposal = evaluate_combo(ctx, tuple(chosen))
        return [proposal for _ in range(max(1, n_samples))]


class RegretKPairRepair(RepairOperator):
    def __init__(self, k: int = 2) -> None:
        self.k = int(k)
        self.name = f"regret{self.k}_repair"

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        remaining = list(ctx.removed_requests)
        chosen: List[PairCandidate] = []

        while remaining:
            best_request: Optional[int] = None
            best_choice: Optional[PairCandidate] = None
            best_regret = -float("inf")
            best_cost = float("inf")

            for request_id in remaining:
                scored: List[Tuple[float, PairCandidate]] = []
                for candidate in ctx.candidate_sets[request_id]:
                    objective = evaluate_partial_combo_objective(ctx, [*chosen, candidate])
                    scored.append((objective, candidate))

                scored.sort(key=lambda item: item[0])
                padded = scored + [scored[-1]] * max(0, self.k - len(scored))
                best_value = padded[0][0]
                regret = sum(padded[min(idx, len(padded) - 1)][0] - best_value for idx in range(1, self.k))

                if regret > best_regret or (abs(regret - best_regret) <= 1e-9 and best_value < best_cost):
                    best_request = request_id
                    best_choice = padded[0][1]
                    best_regret = regret
                    best_cost = best_value

            if best_request is None or best_choice is None:
                raise RuntimeError("Regret repair failed to select a request.")

            chosen.append(best_choice)
            remaining.remove(best_request)

        proposal = evaluate_combo(ctx, tuple(chosen))
        return [proposal for _ in range(max(1, n_samples))]


class ExactMicroSearchRepair(RepairOperator):
    name = "exact_micro_repair"

    def __init__(self, max_states: int = 20_000) -> None:
        self.max_states = int(max_states)

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        best = truth_table(ctx, max_states=self.max_states)[0]
        return [best for _ in range(max(1, n_samples))]


class EntropyBiasedSamplerRepair(RepairOperator):
    name = "entropy_biased_sampler"

    def __init__(self, *, max_states: int = 512, beta: float = 1.0, fallback: Optional[RepairOperator] = None) -> None:
        self.max_states = int(max_states)
        self.beta = float(beta)
        self.fallback = fallback or RegretKPairRepair(2)

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        if ctx.search_space > self.max_states:
            return self.fallback.sample(ctx, rng, n_samples)

        try:
            combos = enumerate_all_combos(ctx, max_states=self.max_states)
        except ValueError:
            return self.fallback.sample(ctx, rng, n_samples)

        energies = np.asarray([combo_approx_energy(ctx, combo) for combo in combos], dtype=float)
        scaled = -self.beta * (energies - float(np.min(energies)))
        scaled -= float(np.max(scaled))
        probabilities = np.exp(scaled)
        probabilities /= float(np.sum(probabilities))

        selected_indices = rng.choices(range(len(combos)), weights=probabilities.tolist(), k=max(1, n_samples))
        return [evaluate_combo(ctx, combos[index]) for index in selected_indices]


# Optional Qiskit helpers for reduced-neighborhood circuit sampling
# ============================================================


def _require_qiskit() -> None:
    if QAOAAnsatz is None or efficient_su2 is None or SparsePauliOp is None:
        raise ImportError(
            "Qiskit is required for QiskitCircuitPairRepair. Install qiskit and, if needed, qiskit-aer or qiskit-ibm-runtime."
        )


def make_params_zeros(params_order) -> np.ndarray:
    return np.zeros(len(params_order), dtype=float)


def make_params_small_random(params_order, rng: np.random.Generator, sigma: float = 0.15) -> np.ndarray:
    return rng.normal(loc=0.0, scale=float(sigma), size=len(params_order)).astype(float)


def make_params_uniform_random(params_order, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 2.0 * np.pi, size=len(params_order)).astype(float)


def make_qaoa_fixed_ref(params_order) -> np.ndarray:
    beta = float(np.pi)
    gamma = float(np.pi / 2.0)
    out = []
    for p in params_order:
        nm = p.name.lower()
        if "beta" in nm or "?" in nm:
            out.append(beta)
        elif "gamma" in nm or "?" in nm:
            out.append(gamma)
        else:
            out.append(0.0)
    return np.asarray(out, dtype=float)


def make_qaoa_layer_ramp(params_order, reps: int) -> np.ndarray:
    beta_vals = np.linspace(np.pi / 8.0, np.pi / 2.0, int(reps), dtype=float)
    gamma_vals = np.linspace(np.pi / 2.0, np.pi / 8.0, int(reps), dtype=float)

    out = []
    beta_idx = 0
    gamma_idx = 0
    for p in params_order:
        nm = p.name.lower()
        if "beta" in nm or "?" in nm:
            out.append(beta_vals[min(beta_idx, len(beta_vals) - 1)])
            beta_idx += 1
        elif "gamma" in nm or "?" in nm:
            out.append(gamma_vals[min(gamma_idx, len(gamma_vals) - 1)])
            gamma_idx += 1
        else:
            out.append(0.0)
    return np.asarray(out, dtype=float)


def make_linear_ramp(params_order) -> np.ndarray:
    n = len(params_order)
    if n == 0:
        return np.zeros(0, dtype=float)
    return np.linspace(0.0, np.pi / 2.0, n, dtype=float)


def make_alternating_pattern(params_order, amplitude: float = np.pi / 4.0) -> np.ndarray:
    n = len(params_order)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        out[i] = amplitude if i % 2 == 0 else -amplitude
    return out


def make_qaoa_init_bank(params_order, reps: int, rng: np.random.Generator) -> List[Tuple[str, np.ndarray]]:
    return [
        ("fixed_ref", make_qaoa_fixed_ref(params_order)),
        ("zeros", make_params_zeros(params_order)),
        ("small_random", make_params_small_random(params_order, rng)),
        ("uniform_random", make_params_uniform_random(params_order, rng)),
        ("layer_ramp", make_qaoa_layer_ramp(params_order, reps=int(reps))),
    ]


def make_vqa_init_bank(params_order, rng: np.random.Generator) -> List[Tuple[str, np.ndarray]]:
    return [
        ("zeros", make_params_zeros(params_order)),
        ("small_random", make_params_small_random(params_order, rng)),
        ("uniform_random", make_params_uniform_random(params_order, rng)),
        ("linear_ramp", make_linear_ramp(params_order)),
        ("alternating", make_alternating_pattern(params_order)),
    ]


def default_make_pass_manager(backend, optimization_level: int, transpile_seed: int):
    _require_qiskit()
    if _qiskit_generate_preset_pass_manager is None:
        raise ImportError("generate_preset_pass_manager is not available in this Qiskit installation.")
    return _qiskit_generate_preset_pass_manager(
        optimization_level=int(optimization_level),
        backend=backend,
        seed_transpiler=int(transpile_seed),
    )


def build_family_circuits_with_init_sweep(
    *,
    backend,
    cost_op,
    num_qubits: int,
    layers: int,
    optimization_level: int,
    transpile_seed: int,
    num_removed: int,
    seed: int,
    make_pass_manager=default_make_pass_manager,
) -> List[Tuple[str, object, List[object], List[Tuple[str, np.ndarray]]]]:
    _require_qiskit()
    pm = make_pass_manager(
        backend,
        optimization_level=int(optimization_level),
        transpile_seed=int(transpile_seed),
    )

    qaoa_logical = QAOAAnsatz(cost_operator=cost_op, reps=int(layers))
    qaoa_logical.measure_all()
    qaoa_params = list(qaoa_logical.parameters)
    qaoa_isa = pm.run(qaoa_logical)
    rng_q = np.random.default_rng(int(10_000 + num_removed * 100 + seed * 10 + layers))
    qaoa_inits = make_qaoa_init_bank(qaoa_params, reps=int(layers), rng=rng_q)

    vqa_logical = efficient_su2(
        num_qubits=int(num_qubits),
        reps=int(layers),
        su2_gates=["rx", "ry", "rz"],
        entanglement="linear",
    )
    vqa_logical.measure_all()
    vqa_params = list(vqa_logical.parameters)
    vqa_isa = pm.run(vqa_logical)
    rng_v = np.random.default_rng(int(20_000 + num_removed * 100 + seed * 10 + layers))
    vqa_inits = make_vqa_init_bank(vqa_params, rng=rng_v)

    return [
        ("qaoa", qaoa_isa, qaoa_params, qaoa_inits),
        ("vqa_su2", vqa_isa, vqa_params, vqa_inits),
    ]


def _next_power_of_two(x: int) -> int:
    out = 1
    while out < int(x):
        out <<= 1
    return out


def _fwht(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    h = 1
    n = len(out)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = out[j]
                y = out[j + h]
                out[j] = x + y
                out[j + h] = x - y
        h *= 2
    out /= float(n)
    return out


def make_diagonal_cost_operator(energies: Sequence[float], pad_penalty: float = 1000.0):
    _require_qiskit()
    energies_arr = np.asarray(list(energies), dtype=float)
    if energies_arr.size == 0:
        raise ValueError("Need at least one energy to build a cost operator.")
    size = _next_power_of_two(len(energies_arr))
    padded = np.full(size, float(np.max(energies_arr)) + float(pad_penalty), dtype=float)
    padded[: len(energies_arr)] = energies_arr
    coeffs = _fwht(padded)
    n = int(round(math.log2(size)))
    terms: List[Tuple[str, float]] = []
    for mask, coeff in enumerate(coeffs):
        if abs(float(coeff)) <= 1e-10:
            continue
        label = "".join("Z" if (mask >> (n - 1 - q)) & 1 else "I" for q in range(n))
        terms.append((label, float(coeff)))
    if not terms:
        terms = [("I" * n, 0.0)]
    return SparsePauliOp.from_list(terms), n, padded


def _format_counts_from_int_dict(int_counts: Dict[int, int], num_qubits: int) -> Dict[str, int]:
    return {format(int(k), f"0{int(num_qubits)}b"): int(v) for k, v in int_counts.items()}


def _extract_counts_from_sampler_payload(payload, num_qubits: int) -> Dict[str, int]:
    if payload is None:
        return {}
    data = getattr(payload, "data", None)
    if data is not None:
        meas = getattr(data, "meas", None)
        if meas is not None:
            if hasattr(meas, "get_counts"):
                counts = meas.get_counts()
                if counts:
                    return {str(k): int(v) for k, v in counts.items()}
            if hasattr(meas, "get_int_counts"):
                counts = meas.get_int_counts()
                if counts:
                    return _format_counts_from_int_dict(counts, num_qubits)
    if hasattr(payload, "quasi_dists") and payload.quasi_dists:
        qd = payload.quasi_dists[0]
        if isinstance(qd, dict):
            return _format_counts_from_int_dict({int(k): int(round(v * 1024)) for k, v in qd.items()}, num_qubits)
    if isinstance(payload, dict):
        if "counts" in payload and isinstance(payload["counts"], dict):
            return {str(k): int(v) for k, v in payload["counts"].items()}
    return {}


def _result_to_counts(result, num_qubits: int, bound_circuit=None) -> Dict[str, int]:
    if result is None:
        return {}
    if hasattr(result, "get_counts"):
        try:
            counts = result.get_counts(bound_circuit) if bound_circuit is not None else result.get_counts()
            if isinstance(counts, dict) and counts:
                return {str(k): int(v) for k, v in counts.items()}
        except Exception:
            pass
    try:
        first = result[0]
        counts = _extract_counts_from_sampler_payload(first, num_qubits)
        if counts:
            return counts
    except Exception:
        pass
    counts = _extract_counts_from_sampler_payload(result, num_qubits)
    if counts:
        return counts
    return {}


def _normalize_counts(counts: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, val in counts.items():
        bitstr = str(key).replace(" ", "")
        if bitstr.startswith("0x"):
            bitstr = format(int(bitstr, 16), "b")
        out[bitstr] = out.get(bitstr, 0) + int(val)
    return out


def normalized_counts_entropy(counts: Dict[str, int], *, support_tol: float = 1e-6) -> float:
    total = float(sum(max(0, int(value)) for value in counts.values()))
    if total <= 0.0:
        return 0.0

    probs = np.asarray([max(0, int(value)) / total for value in counts.values()], dtype=float)
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0

    entropy = -float(np.sum(probs * np.log(probs + 1e-15)))
    effective_support = max(2, int(np.sum(probs > float(support_tol))))
    return float(min(1.0, max(0.0, entropy / math.log(effective_support))))


def _clip_unit_interval(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))




class QiskitCircuitPairRepair(RepairOperator):
    """
    Real Qiskit-backed reduced-neighborhood sampler.

    The reduced neighborhood is enumerated into a candidate pool, then indexed by
    computational basis states. A diagonal cost Hamiltonian is built from the
    approximate reduced-neighborhood energies, and either QAOAAnsatz or
    EfficientSU2 is executed to sample basis states.

    If target_entropy is set, the init bank is probed with a small shot budget
    and the initialization whose measured entropy is closest to the target is
    used for the final repair sampling call.
    """

    def __init__(
        self,
        *,
        backend,
        family: str = "qaoa",
        layers: int = 1,
        shots: int = 256,
        optimization_level: int = 1,
        transpile_seed: int = 1234,
        make_pass_manager=default_make_pass_manager,
        sampler=None,
        pad_penalty: float = 1000.0,
        max_states: int = 256,
        fallback_operator: Optional[RepairOperator] = None,
        aggregate_inits: bool = True,
        target_entropy: Optional[float] = None,
        entropy_probe_shots: int = 64,
    ):
        self.backend = backend
        self.family = str(family)
        self.layers = int(layers)
        self.shots = int(shots)
        self.optimization_level = int(optimization_level)
        self.transpile_seed = int(transpile_seed)
        self.make_pass_manager = make_pass_manager
        self.sampler = sampler
        self.pad_penalty = float(pad_penalty)
        self.max_states = int(max_states)
        self.aggregate_inits = bool(aggregate_inits)
        self.target_entropy = None if target_entropy is None else _clip_unit_interval(float(target_entropy))
        self.entropy_probe_shots = max(1, int(entropy_probe_shots))
        self.fallback_operator = fallback_operator if fallback_operator is not None else RegretKPairRepair(2)
        self.last_error: Optional[Exception] = None
        self.last_selected_init: Optional[str] = None
        self.last_selected_init_entropy: Optional[float] = None
        self.last_init_entropy_scores: Dict[str, float] = {}
        self.name = f"qiskit_{self.family}_repair"

    def _candidate_pool(self, ctx: RepairContext) -> List[Tuple[PairCandidate, ...]]:
        if ctx.search_space <= self.max_states:
            return enumerate_all_combos(ctx, max_states=self.max_states)
        raise ValueError(
            f"QiskitCircuitPairRepair requires full pool enumeration within max_states. Got {ctx.search_space} > {self.max_states}."
        )

    def _run_bound_circuit(self, bound_circuit, num_qubits: int, shots: int) -> Dict[str, int]:
        run_shots = int(shots)
        if self.sampler is not None:
            try:
                job = self.sampler.run([bound_circuit], shots=run_shots)
            except TypeError:
                job = self.sampler.run([(bound_circuit,)], shots=run_shots)
            result = job.result()
            return _normalize_counts(_result_to_counts(result, num_qubits))
        if self.backend is None:
            raise ValueError("Either backend or sampler must be provided for QiskitCircuitPairRepair.")
        job = self.backend.run(bound_circuit, shots=run_shots)
        result = job.result()
        return _normalize_counts(_result_to_counts(result, num_qubits, bound_circuit=bound_circuit))

    def set_target_entropy(self, target_entropy: Optional[float]) -> None:
        self.target_entropy = None if target_entropy is None else _clip_unit_interval(float(target_entropy))

    def _bind_circuit(self, circuit, params_order: List[object], init_values: np.ndarray):
        return circuit.assign_parameters(
            {param: float(value) for param, value in zip(params_order, init_values)},
            inplace=False,
        )

    def _select_entropy_matched_init(
        self,
        *,
        circuit,
        params_order: List[object],
        init_bank: List[Tuple[str, np.ndarray]],
        num_qubits: int,
        target_entropy: float,
    ) -> Tuple[str, np.ndarray]:
        probe_shots = max(1, min(int(self.entropy_probe_shots), int(self.shots)))
        best_item: Optional[Tuple[str, np.ndarray]] = None
        best_score = float("inf")
        self.last_init_entropy_scores = {}

        for init_kind, init_values in init_bank:
            bound = self._bind_circuit(circuit, params_order, init_values)
            counts = self._run_bound_circuit(bound, num_qubits, shots=probe_shots)
            measured_entropy = normalized_counts_entropy(counts)
            self.last_init_entropy_scores[init_kind] = measured_entropy
            score = abs(measured_entropy - float(target_entropy))
            if score < best_score:
                best_score = score
                best_item = (init_kind, init_values)

        if best_item is None:
            return init_bank[0]

        self.last_selected_init = best_item[0]
        self.last_selected_init_entropy = self.last_init_entropy_scores.get(best_item[0])
        return best_item

    def sample(self, ctx: RepairContext, rng: random.Random, n_samples: int) -> List[RepairProposal]:
        try:
            _require_qiskit()
            if self.backend is None and self.sampler is None:
                return self.fallback_operator.sample(ctx, rng, n_samples)

            pool = self._candidate_pool(ctx)
            if not pool:
                return self.fallback_operator.sample(ctx, rng, n_samples)

            energies = [combo_approx_energy(ctx, combo) for combo in pool]
            cost_op, num_qubits, padded_energies = make_diagonal_cost_operator(energies, pad_penalty=self.pad_penalty)
            backend_for_transpile = self.backend
            if backend_for_transpile is None and hasattr(self.sampler, "backend"):
                backend_for_transpile = getattr(self.sampler, "backend")
            if backend_for_transpile is None:
                return self.fallback_operator.sample(ctx, rng, n_samples)

            families = build_family_circuits_with_init_sweep(
                backend=backend_for_transpile,
                cost_op=cost_op,
                num_qubits=num_qubits,
                layers=self.layers,
                optimization_level=self.optimization_level,
                transpile_seed=self.transpile_seed,
                num_removed=len(ctx.removed_requests),
                seed=rng.randint(0, 10**9),
                make_pass_manager=self.make_pass_manager,
            )
            family_map = {fam: (circ, params, init_bank) for fam, circ, params, init_bank in families}
            if self.family not in family_map:
                raise ValueError(f"Unknown family '{self.family}'. Available: {sorted(family_map)}")
            circuit, params_order, init_bank = family_map[self.family]

            aggregate_counts: Dict[str, int] = {}
            if self.target_entropy is not None:
                inits_to_use = [
                    self._select_entropy_matched_init(
                        circuit=circuit,
                        params_order=params_order,
                        init_bank=init_bank,
                        num_qubits=num_qubits,
                        target_entropy=self.target_entropy,
                    )
                ]
            else:
                self.last_selected_init = None
                self.last_selected_init_entropy = None
                self.last_init_entropy_scores = {}
                inits_to_use = init_bank if self.aggregate_inits else [init_bank[0]]

            per_init_shots = max(1, int(self.shots // max(1, len(inits_to_use))))

            for init_kind, init_values in inits_to_use:
                bound = self._bind_circuit(circuit, params_order, init_values)
                counts = self._run_bound_circuit(bound, num_qubits, shots=per_init_shots)
                for bitstr, cnt in counts.items():
                    aggregate_counts[bitstr] = aggregate_counts.get(bitstr, 0) + int(cnt)

            if not aggregate_counts:
                return self.fallback_operator.sample(ctx, rng, n_samples)

            expanded: List[RepairProposal] = []
            ordered_items = sorted(aggregate_counts.items(), key=lambda kv: kv[1], reverse=True)
            for bitstr, cnt in ordered_items:
                idx = int(bitstr, 2)
                if idx >= len(pool):
                    continue
                proposal = evaluate_combo(ctx, pool[idx])
                reps = min(int(cnt), max(1, n_samples - len(expanded)))
                expanded.extend([proposal] * reps)
                if len(expanded) >= n_samples:
                    break

            if not expanded:
                return self.fallback_operator.sample(ctx, rng, n_samples)
            if len(expanded) < n_samples:
                base_len = len(expanded)
                while len(expanded) < n_samples:
                    expanded.append(expanded[len(expanded) % base_len])
            return expanded[:n_samples]
        except Exception as exc:
            self.last_error = exc
            return self.fallback_operator.sample(ctx, rng, n_samples)


def build_quantum_repair_operator(
    *,
    family: str = "qaoa",
    backend=None,
    sampler=None,
    layers: int = 1,
    shots: int = 256,
    max_states: int = 256,
    optimization_level: int = 1,
    transpile_seed: int = 1234,
    fallback_operator: Optional[RepairOperator] = None,
    target_entropy: Optional[float] = None,
    entropy_probe_shots: int = 64,
) -> RepairOperator:
    return QiskitCircuitPairRepair(
        backend=backend,
        sampler=sampler,
        family=family,
        layers=layers,
        shots=shots,
        optimization_level=optimization_level,
        transpile_seed=transpile_seed,
        max_states=max_states,
        fallback_operator=fallback_operator or RegretKPairRepair(2),
        target_entropy=target_entropy,
        entropy_probe_shots=entropy_probe_shots,
    )

def build_classical_destroy_pool() -> List[DestroyOperator]:
    return [RandomRequestRemoval(), ShawRequestRemoval(), WorstRequestRemoval()]


def build_classical_repair_pool() -> List[RepairOperator]:
    return [GreedyPairRepair(), RegretKPairRepair(2), RegretKPairRepair(3)]
