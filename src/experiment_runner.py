from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from alns import ALNSRunner
from entropy import EntropyFeatureExtractor
from pdptw import capacity_utilization, generate_pdptw_instance, objective_gap, total_node_count
from policy import CircuitSelector, EntropyAwareRepairOperator, EntropyRepairPolicy, QuantumRepairPortfolio

try:
    from rl_training import load_bandit_policy
except Exception:  # pragma: no cover - optional deployment helper
    load_bandit_policy = None

try:
    from dqn_policy import load_dqn_policy
except Exception:  # pragma: no cover - optional deployment helper
    load_dqn_policy = None

try:
    from hardware_noise import CircuitNoiseContext, EmpiricalHardwareNoiseModel
except Exception:  # pragma: no cover - optional deployment helper
    CircuitNoiseContext = None
    EmpiricalHardwareNoiseModel = None

from repair import (
    GreedyPairRepair,
    RegretKPairRepair,
    build_classical_destroy_pool,
    build_classical_repair_pool,
    build_quantum_repair_operator,
)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def stable_int_hash(value: str, *, modulo: int = 999_983) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(modulo)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_save_convergence_plot(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped: Dict[str, Dict[int, List[float]]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        iteration = int(row.get("iteration", 0))
        grouped.setdefault(method, {}).setdefault(iteration, []).append(float(row.get("best_objective", np.nan)))

    if not grouped:
        return

    ensure_dir(path.parent)
    plt.figure(figsize=(8, 5))
    for method, iter_values in sorted(grouped.items()):
        xs = sorted(iter_values)
        ys = [float(np.nanmean(iter_values[x])) for x in xs]
        plt.plot(xs, ys, label=method)

    plt.xlabel("ALNS iteration")
    plt.ylabel("Mean best objective")
    plt.title("Convergence comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def load_noise_model(path: Optional[Path]):
    if path is None:
        return None
    if EmpiricalHardwareNoiseModel is None:
        raise RuntimeError("hardware_noise.py could not be imported, so --hardware-noise-model cannot be used.")
    return EmpiricalHardwareNoiseModel.load(Path(path))


def predicted_context_noise_score(
    *,
    hardware_noise_model,
    fallback_noise_score: float,
    remove_count: int,
    candidate_cap: int,
    qiskit_shots: int,
) -> float:
    if hardware_noise_model is None:
        return float(fallback_noise_score)
    return float(
        hardware_noise_model.predict_min_noise_score(
            num_removed=int(remove_count),
            candidates_per_removed=int(candidate_cap),
            shots=int(qiskit_shots),
            families=("qaoa", "vqa_su2"),
            layers=(1, 2),
        )
    )


def build_entropy_repair(
    *,
    noise_score: float,
    quantum_operator=None,
    circuit_shots: int = 256,
    policy_model: Path | None = None,
    dqn_model: Path | None = None,
    min_quantum_observations: int = 5,
    quantum_eval_ratio: float = 1.0,
) -> EntropyAwareRepairOperator:
    extractor = EntropyFeatureExtractor(search_space_reference=10_000)

    if policy_model is not None:
        if load_bandit_policy is None:
            raise RuntimeError("rl_training.py could not be imported, so --policy-model cannot be used.")
        policy = load_bandit_policy(
            Path(policy_model),
            exact_max_states=128,
            min_quantum_observations=min_quantum_observations,
        )
    elif dqn_model is not None:
        if load_dqn_policy is None:
            raise RuntimeError("dqn_policy.py could not be imported, so --dqn-model cannot be used.")
        policy = load_dqn_policy(
            Path(dqn_model),
            exact_max_states=128,
            min_quantum_observations=min_quantum_observations,
        )
    else:
        policy = EntropyRepairPolicy(
            exact_max_states=128,
            quantum_min_entropy=0.30,
            quantum_max_entropy=0.90,
            quantum_min_conflict_density=0.03,
            quantum_max_noise=0.50,
            quantum_max_removed=5,
            circuit_selector=CircuitSelector(shots=circuit_shots),
        )

    return EntropyAwareRepairOperator(
        extractor=extractor,
        policy=policy,
        quantum_operator=quantum_operator,
        search_state={"stagnation": 0.0},
        noise_state={"noise_score": float(noise_score)},
        quantum_eval_ratio=quantum_eval_ratio,
    )


def build_optional_quantum_portfolio(
    *,
    use_qiskit_aer: bool,
    qiskit_shots: int,
    qiskit_max_states: int,
):
    if not use_qiskit_aer:
        return None

    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise RuntimeError("Install qiskit-aer or run without --use-qiskit-aer.") from exc

    backend = AerSimulator(seed_simulator=1234)
    operators = {}
    for family in ("qaoa", "vqa_su2"):
        for layers in (1, 2):
            operator = build_quantum_repair_operator(
                family=family,
                backend=backend,
                layers=layers,
                shots=qiskit_shots,
                max_states=qiskit_max_states,
                transpile_seed=1234 + 31 * layers + stable_int_hash(family, modulo=997),
            )
            operators[f"{family}:L{layers}:S{qiskit_shots}"] = operator
            operators[f"{family}:L{layers}"] = operator
            operators.setdefault(family, operator)

    return QuantumRepairPortfolio(operators)


def build_fixed_qc_baseline_pools(
    *,
    use_qiskit_aer: bool,
    qiskit_shots: int,
    qiskit_max_states: int,
    include_fixed_qc_baselines: bool,
) -> Dict[str, List[object]]:
    """Build full-ALNS Qiskit baselines from the original monolithic scaffold.

    These baselines intentionally use quantum repair throughout the ALNS run:
    - qaoa_only_alns: every repair call uses QAOA sampling;
    - su2_only_alns: every repair call uses EfficientSU2 sampling;
    - hybrid_qc_roulette_alns: ALNS roulette chooses among classical and QC repairs.

    They are only enabled for real Qiskit/Aer runs because proxy baselines would
    blur the fixed-QC comparison.
    """
    if not include_fixed_qc_baselines or not use_qiskit_aer:
        return {}

    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise RuntimeError("Install qiskit-aer or disable --include-fixed-qc-baselines.") from exc

    backend = AerSimulator(seed_simulator=4321)
    qaoa = build_quantum_repair_operator(
        family="qaoa",
        backend=backend,
        layers=1,
        shots=qiskit_shots,
        max_states=qiskit_max_states,
        transpile_seed=2027,
        target_entropy=None,
    )
    su2 = build_quantum_repair_operator(
        family="vqa_su2",
        backend=backend,
        layers=1,
        shots=qiskit_shots,
        max_states=qiskit_max_states,
        transpile_seed=2063,
        target_entropy=None,
    )

    return {
        "qaoa_only_alns": [qaoa],
        "su2_only_alns": [su2],
        "hybrid_qc_roulette_alns": [
            GreedyPairRepair(),
            RegretKPairRepair(2),
            RegretKPairRepair(3),
            qaoa,
            su2,
        ],
    }


def run_suite(
    *,
    out_dir: Path,
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
    policy_model: Path | None = None,
    dqn_model: Path | None = None,
    min_quantum_observations: int = 5,
    quantum_eval_ratio: float = 1.0,
    include_fixed_qc_baselines: bool = True,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    convergence_rows: List[Dict[str, object]] = []
    destroy_pool = build_classical_destroy_pool()
    quantum_operator = build_optional_quantum_portfolio(
        use_qiskit_aer=use_qiskit_aer,
        qiskit_shots=qiskit_shots,
        qiskit_max_states=qiskit_max_states,
    )
    hardware_noise_model = load_noise_model(hardware_noise_model_path)
    fixed_qc_baselines = build_fixed_qc_baseline_pools(
        use_qiskit_aer=use_qiskit_aer,
        qiskit_shots=qiskit_shots,
        qiskit_max_states=qiskit_max_states,
        include_fixed_qc_baselines=include_fixed_qc_baselines,
    )

    for remove_count in remove_counts:
        for candidate_cap in candidate_caps:
            effective_noise_score = predicted_context_noise_score(
                hardware_noise_model=hardware_noise_model,
                fallback_noise_score=noise_score,
                remove_count=int(remove_count),
                candidate_cap=int(candidate_cap),
                qiskit_shots=int(qiskit_shots),
            )

            for n_requests in sizes:
                for seed in seeds:
                    inst = generate_pdptw_instance(
                        n_requests,
                        k_vehicles,
                        seed=seed,
                        tw_tightness=tw_tightness,
                        capacity_slack=capacity_slack,
                    )

                    method_repair_pools = {
                        "regret2": [RegretKPairRepair(2)],
                        "classical_alns": build_classical_repair_pool(),
                    }
                    if policy_model is None and dqn_model is None:
                        method_repair_pools["entropy_alns"] = [
                            GreedyPairRepair(),
                            RegretKPairRepair(2),
                            build_entropy_repair(
                                noise_score=effective_noise_score,
                                quantum_operator=quantum_operator,
                                circuit_shots=qiskit_shots,
                                min_quantum_observations=min_quantum_observations,
                                quantum_eval_ratio=quantum_eval_ratio,
                            ),
                        ]
                    if policy_model is not None:
                        method_repair_pools["bandit_alns"] = [
                            GreedyPairRepair(),
                            RegretKPairRepair(2),
                            build_entropy_repair(
                                noise_score=effective_noise_score,
                                quantum_operator=quantum_operator,
                                circuit_shots=qiskit_shots,
                                policy_model=policy_model,
                                min_quantum_observations=min_quantum_observations,
                                quantum_eval_ratio=quantum_eval_ratio,
                            ),
                        ]
                    if dqn_model is not None:
                        method_repair_pools["dqn_alns"] = [
                            GreedyPairRepair(),
                            RegretKPairRepair(2),
                            build_entropy_repair(
                                noise_score=effective_noise_score,
                                quantum_operator=quantum_operator,
                                circuit_shots=qiskit_shots,
                                dqn_model=dqn_model,
                                min_quantum_observations=min_quantum_observations,
                                quantum_eval_ratio=quantum_eval_ratio,
                            ),
                        ]
                    method_repair_pools.update(fixed_qc_baselines)

                    per_method = {}
                    for method_name, repair_pool in method_repair_pools.items():
                        runner = ALNSRunner(destroy_pool, repair_pool)
                        _, best_eval, trace = runner.run(
                            inst,
                            seed=10_000 * seed + stable_int_hash(f"{method_name}:{remove_count}:{candidate_cap}", modulo=999),
                            iterations=iterations,
                            remove_count=int(remove_count),
                            candidate_cap=int(candidate_cap),
                            samples_per_call=samples_per_call,
                            method_name=method_name,
                            noise_score=effective_noise_score,
                        )
                        per_method[method_name] = (best_eval, trace)
                        for idx, iter_idx in enumerate(trace.iter_idx):
                            convergence_rows.append(
                                {
                                    "method": method_name,
                                    "iteration": int(iter_idx),
                                    "best_objective": float(trace.best_objective[idx]),
                                    "current_objective": float(trace.current_objective[idx]),
                                    "elapsed_sec": float(trace.elapsed_sec[idx]),
                                    "quantum_calls": int(trace.quantum_calls[idx]),
                                    "n_requests": int(n_requests),
                                    "seed": int(seed),
                                    "remove_count": int(remove_count),
                                    "candidate_cap": int(candidate_cap),
                                    "tw_tightness": float(tw_tightness),
                                    "capacity_slack": float(capacity_slack),
                                }
                            )

                    reference = min(best_eval.objective for best_eval, _ in per_method.values())
                    for method_name, (best_eval, trace) in per_method.items():
                        rows.append(
                            {
                                "n_requests": n_requests,
                                "total_nodes": total_node_count(inst),
                                "capacity_utilization": capacity_utilization(inst),
                                "k_vehicles": k_vehicles,
                                "seed": seed,
                                "method": method_name,
                                "remove_count": int(remove_count),
                                "candidate_cap": int(candidate_cap),
                                "search_space_cap": int(candidate_cap) ** int(remove_count),
                                "final_objective": best_eval.objective,
                                "gap_to_seed_best": objective_gap(best_eval.objective, reference),
                                "feasible": best_eval.feasible,
                                "elapsed_sec": trace.elapsed_sec[-1] if trace.elapsed_sec else np.nan,
                                "quantum_calls": trace.quantum_calls[-1] if trace.quantum_calls else 0,
                                "tw_tightness": tw_tightness,
                                "capacity_slack": capacity_slack,
                                "noise_score": effective_noise_score,
                                "noise_source": str(hardware_noise_model_path) if hardware_noise_model_path is not None else "constant",
                                "policy_model": str(policy_model) if policy_model is not None else "",
                                "dqn_model": str(dqn_model) if dqn_model is not None else "",
                                "quantum_eval_ratio": quantum_eval_ratio,
                                "fixed_qc_baseline": method_name in fixed_qc_baselines,
                            }
                        )

    save_csv(out_dir / "summary.csv", rows)
    save_csv(out_dir / "convergence.csv", convergence_rows)
    maybe_save_convergence_plot(out_dir / "convergence_comparison.png", convergence_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final entropy/noise-aware PDPTW ALNS experiments.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/entropy_pdptw"))
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--sizes", type=str, default="20,30")
    parser.add_argument("--k-vehicles", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--remove-counts",
        type=str,
        default="2,3,4,5",
        help="Comma-separated removed-request counts. Final-run default: 2,3,4,5.",
    )
    parser.add_argument(
        "--candidate-caps",
        type=str,
        default="2,3,4",
        help="Comma-separated candidate caps per removed request. Final-run default: 2,3,4.",
    )
    parser.add_argument("--samples-per-call", type=int, default=32)
    parser.add_argument("--tw-tightness", type=float, default=0.5)
    parser.add_argument("--capacity-slack", type=float, default=0.30)
    parser.add_argument("--noise-score", type=float, default=0.15)
    parser.add_argument(
        "--hardware-noise-model",
        type=Path,
        default=None,
        help="Optional JSON model from hardware_noise.py. Overrides constant --noise-score by setting-specific predictions.",
    )
    parser.add_argument("--use-qiskit-aer", action="store_true")
    parser.add_argument("--qiskit-shots", type=int, default=256)
    parser.add_argument(
        "--qiskit-max-states",
        type=int,
        default=1024,
        help="Default 1024 supports the final grid max 4^5 reduced-neighbourhood states.",
    )
    parser.add_argument(
        "--policy-model",
        type=Path,
        default=None,
        help="Optional rl_training.py JSON model for hierarchical contextual-bandit repair/circuit selection.",
    )
    parser.add_argument(
        "--dqn-model",
        type=Path,
        default=None,
        help="Optional dqn_train.py JSON model for Double-DQN long-horizon control.",
    )
    parser.add_argument(
        "--min-quantum-observations",
        type=int,
        default=5,
        help="Minimum training rows required per quantum action before the learned policy may select it.",
    )
    parser.add_argument(
        "--quantum-eval-ratio",
        type=float,
        default=1.0,
        help="Probability of allowing a policy-selected quantum repair during experiment deployment. "
        "Use <1.0 to limit quantum calls under a runtime budget.",
    )
    parser.add_argument(
        "--include-fixed-qc-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include qaoa_only_alns, su2_only_alns, and hybrid_qc_roulette_alns when --use-qiskit-aer is enabled.",
    )
    args = parser.parse_args()

    rows = run_suite(
        out_dir=args.out_dir,
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
        policy_model=args.policy_model,
        dqn_model=args.dqn_model,
        min_quantum_observations=args.min_quantum_observations,
        quantum_eval_ratio=args.quantum_eval_ratio,
        include_fixed_qc_baselines=args.include_fixed_qc_baselines,
    )

    print(f"Wrote {len(rows)} rows to {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
