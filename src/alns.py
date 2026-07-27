from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pdptw import PDPTWInstance, RunTrace, SolutionEval, deep_copy_routes, evaluate_solution, greedy_initial_solution
from policy import EntropyAwareRepairOperator
from repair import DestroyOperator, RepairOperator, build_repair_context


@dataclass
class OperatorScoreState:
    weights: Dict[str, float]
    scores: Dict[str, float]
    counts: Dict[str, int]

    @classmethod
    def build(cls, names: Sequence[str], init_weight: float = 1.0) -> "OperatorScoreState":
        return cls(
            weights={name: float(init_weight) for name in names},
            scores={name: 0.0 for name in names},
            counts={name: 0 for name in names},
        )


def roulette_choice(names: Sequence[str], weights: Dict[str, float], rng: random.Random) -> str:
    safe_weights = [max(1e-9, weights[name]) for name in names]
    return rng.choices(list(names), weights=safe_weights, k=1)[0]


def reward_value(candidate_obj: float, current_obj: float, best_obj: float, accepted: bool) -> float:
    if candidate_obj < best_obj - 1e-9:
        return 10.0
    if candidate_obj < current_obj - 1e-9:
        return 6.0
    if accepted:
        return 2.0
    return 0.0


def repair_operator_uses_quantum(operator: RepairOperator) -> bool:
    """Return True when the selected repair operator attempted quantum sampling."""
    name = str(getattr(operator, "name", "")).lower()
    return bool(getattr(operator, "is_quantum", False)) or name.startswith("qiskit_") or name == "quantum_repair"


class ALNSRunner:
    def __init__(
        self,
        destroy_operators: Sequence[DestroyOperator],
        repair_operators: Sequence[RepairOperator],
        *,
        reaction_factor: float = 0.2,
        segment_length: int = 25,
    ) -> None:
        self.destroy_ops = list(destroy_operators)
        self.repair_ops = list(repair_operators)
        self.destroy_state = OperatorScoreState.build([operator.name for operator in self.destroy_ops])
        self.repair_state = OperatorScoreState.build([operator.name for operator in self.repair_ops])
        self.reaction_factor = float(reaction_factor)
        self.segment_length = int(segment_length)

    def _update_weights(self) -> None:
        for state in (self.destroy_state, self.repair_state):
            for name in list(state.weights):
                if state.counts[name] > 0:
                    average = state.scores[name] / state.counts[name]
                    state.weights[name] = (1.0 - self.reaction_factor) * state.weights[name] + self.reaction_factor * average
                state.scores[name] = 0.0
                state.counts[name] = 0

    def run(
        self,
        inst: PDPTWInstance,
        *,
        seed: int,
        iterations: int,
        remove_count: int,
        candidate_cap: int,
        samples_per_call: int,
        start_temperature: float = 10.0,
        cooling: float = 0.995,
        method_name: str = "entropy_alns",
        noise_score: float = 0.0,
        step_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Tuple[List[List[int]], SolutionEval, RunTrace]:
        rng = random.Random(seed)
        current_routes = greedy_initial_solution(inst)
        current_eval = evaluate_solution(inst, current_routes)
        best_routes = deep_copy_routes(current_routes)
        best_eval = current_eval

        trace = RunTrace(method_name, seed, f"R{inst.n_requests}_K{inst.k_vehicles}")
        temperature = float(start_temperature)
        started_at = time.perf_counter()
        no_improve_iters = 0
        quantum_calls = 0
        accepted_last_step = False
        improved_last_step = False

        for iteration in range(iterations):
            destroy_name = roulette_choice([operator.name for operator in self.destroy_ops], self.destroy_state.weights, rng)
            repair_name = roulette_choice([operator.name for operator in self.repair_ops], self.repair_state.weights, rng)
            destroy_operator = next(operator for operator in self.destroy_ops if operator.name == destroy_name)
            repair_operator = next(operator for operator in self.repair_ops if operator.name == repair_name)

            partial_routes, removed_requests = destroy_operator.destroy(inst, current_routes, remove_count, rng)
            ctx = build_repair_context(inst, partial_routes, removed_requests, candidate_cap)

            iteration_fraction = iteration / max(1, iterations - 1) if iterations > 1 else 1.0
            temperature_fraction = temperature / max(1e-9, float(start_temperature))
            if isinstance(repair_operator, EntropyAwareRepairOperator):
                repair_operator.search_state["stagnation"] = min(1.0, no_improve_iters / max(1, iterations))
                repair_operator.search_state["iteration_fraction"] = float(iteration_fraction)
                repair_operator.search_state["accepted_last_step"] = float(accepted_last_step)
                repair_operator.search_state["improved_last_step"] = float(improved_last_step)
                repair_operator.search_state["temperature_fraction"] = float(temperature_fraction)
                repair_operator.noise_state["noise_score"] = float(noise_score)

            proposal = repair_operator.best_repair(ctx, rng, samples_per_call)
            new_routes = proposal.routes
            new_eval = evaluate_solution(inst, new_routes)

            previous_current_objective = current_eval.objective
            accepted = new_eval.objective <= current_eval.objective
            if not accepted:
                delta = new_eval.objective - current_eval.objective
                accepted = rng.random() < math.exp(-delta / max(1e-9, temperature))

            if accepted:
                current_routes = new_routes
                current_eval = new_eval

            reward = reward_value(new_eval.objective, previous_current_objective, best_eval.objective, accepted)
            self.destroy_state.scores[destroy_name] += reward
            self.destroy_state.counts[destroy_name] += 1
            self.repair_state.scores[repair_name] += reward
            self.repair_state.counts[repair_name] += 1

            improved_best = new_eval.objective < best_eval.objective
            if improved_best:
                best_routes = deep_copy_routes(new_routes)
                best_eval = new_eval
                no_improve_iters = 0
            else:
                no_improve_iters += 1

            quantum_used = repair_operator_uses_quantum(repair_operator)
            if isinstance(repair_operator, EntropyAwareRepairOperator):
                quantum_used = bool(repair_operator.last_decision and repair_operator.last_decision.use_quantum)
            if quantum_used:
                quantum_calls += 1

            if (iteration + 1) % self.segment_length == 0:
                self._update_weights()

            trace.iter_idx.append(iteration)
            trace.best_objective.append(best_eval.objective)
            trace.current_objective.append(current_eval.objective)
            trace.elapsed_sec.append(time.perf_counter() - started_at)
            trace.quantum_calls.append(quantum_calls)

            dqn_reward = float(reward)
            if accepted:
                dqn_reward += 0.5
            if improved_best:
                dqn_reward += 2.0
            if quantum_used:
                dqn_reward -= 0.1

            if step_callback is not None:
                step_callback(
                    {
                        "iteration": iteration,
                        "iteration_fraction": float(iteration_fraction),
                        "temperature": float(temperature),
                        "temperature_fraction": float(temperature_fraction),
                        "destroy_name": destroy_name,
                        "repair_name": repair_name,
                        "ctx_search_space": int(ctx.search_space),
                        "ctx_removed_count": len(ctx.removed_requests),
                        "candidate_cap": int(candidate_cap),
                        "reward": float(reward),
                        "dqn_reward": float(dqn_reward),
                        "accepted": bool(accepted),
                        "improved_best": bool(improved_best),
                        "best_objective": float(best_eval.objective),
                        "current_objective": float(current_eval.objective),
                        "new_objective": float(new_eval.objective),
                        "quantum_used": bool(quantum_used),
                        "quantum_calls": int(quantum_calls),
                        "noise_score": float(noise_score),
                        "done": bool(iteration + 1 >= iterations),
                        "features": getattr(repair_operator, "last_features", None),
                        "decision": getattr(repair_operator, "last_decision", None),
                    }
                )

            accepted_last_step = bool(accepted)
            improved_last_step = bool(improved_best)
            temperature *= float(cooling)

        return best_routes, best_eval, trace
