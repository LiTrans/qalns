from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


NOISE_FEATURE_NAMES = [
    "bias",
    "is_qaoa",
    "is_vqa_su2",
    "layers",
    "log_shots",
    "num_removed",
    "candidates_per_removed",
    "num_vars",
    "circ_depth",
    "circ_twoq_ops",
    "optimization_level",
]


@dataclass(frozen=True)
class HardwareNoisePrediction:
    predicted_tvd: float
    predicted_entropy_gap: float
    predicted_feasible_loss: float
    predicted_optimum_loss: float
    predicted_runtime_sec: float
    noise_score: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "predicted_tvd": self.predicted_tvd,
            "predicted_entropy_gap": self.predicted_entropy_gap,
            "predicted_feasible_loss": self.predicted_feasible_loss,
            "predicted_optimum_loss": self.predicted_optimum_loss,
            "predicted_runtime_sec": self.predicted_runtime_sec,
            "noise_score": self.noise_score,
        }


@dataclass(frozen=True)
class CircuitNoiseContext:
    family: str
    layers: int
    shots: int
    num_removed: int
    candidates_per_removed: int
    num_vars: Optional[int] = None
    circ_depth: Optional[float] = None
    circ_twoq_ops: Optional[float] = None
    optimization_level: int = 3


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return float(default)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _family_key(family: str) -> str:
    text = str(family).strip().lower()
    if text in {"su2", "efficient_su2", "efficient-su2"}:
        return "vqa_su2"
    return text


def _percentile_scale(values: np.ndarray, percentile: float = 95.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    scale = float(np.percentile(np.abs(finite), percentile))
    return max(1e-9, scale)


class EmpiricalHardwareNoiseModel:
    """Small ridge model with median circuit-profile fallback.

    The model stores separate linear coefficients for five targets:
    TVD, entropy gap, feasible-probability loss, optimum-probability loss, and
    hardware runtime. The final scalar ``noise_score`` is a weighted normalized
    combination of those targets.
    """

    def __init__(
        self,
        *,
        feature_names: Sequence[str] = NOISE_FEATURE_NAMES,
        ridge_alpha: float = 1.0,
        target_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.ridge_alpha = float(ridge_alpha)
        self.target_weights = dict(
            target_weights
            or {
                "predicted_tvd": 0.40,
                "predicted_entropy_gap": 0.20,
                "predicted_feasible_loss": 0.20,
                "predicted_optimum_loss": 0.15,
                "predicted_runtime_sec": 0.05,
            }
        )
        self.feature_mean: List[float] = []
        self.feature_std: List[float] = []
        self.coefficients: Dict[str, List[float]] = {}
        self.target_scales: Dict[str, float] = {}
        self.profile_medians: Dict[str, Dict[str, float]] = {}
        self.default_profile: Dict[str, float] = {
            "circ_depth": 1.0,
            "circ_twoq_ops": 0.0,
            "num_vars": 1.0,
        }
        self.rows_seen = 0
        self.backend_names: List[str] = []

    @staticmethod
    def load_csv_rows(path: Path) -> List[Dict[str, str]]:
        with Path(path).open("r", newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    @staticmethod
    def _row_targets(row: Mapping[str, object]) -> Dict[str, float]:
        tvd = _clip01(_safe_float(row.get("tvd_hw_vs_aer"), 0.0))
        entropy_gap = abs(_safe_float(row.get("hw_entropy"), 0.0) - _safe_float(row.get("aer_entropy"), 0.0))
        feasible_loss = max(0.0, _safe_float(row.get("aer_feasible_prob"), 0.0) - _safe_float(row.get("hw_feasible_prob"), 0.0))
        optimum_loss = max(0.0, _safe_float(row.get("aer_p_exact_optimum"), 0.0) - _safe_float(row.get("hw_p_exact_optimum"), 0.0))
        runtime = max(0.0, _safe_float(row.get("hw_wall_sec"), 0.0))
        return {
            "predicted_tvd": tvd,
            "predicted_entropy_gap": entropy_gap,
            "predicted_feasible_loss": feasible_loss,
            "predicted_optimum_loss": optimum_loss,
            "predicted_runtime_sec": runtime,
        }

    @staticmethod
    def _profile_key(
        *,
        family: str,
        layers: int,
        shots: int,
        num_removed: int,
        candidates_per_removed: int,
    ) -> str:
        return (
            f"{_family_key(family)}|L{int(layers)}|S{int(shots)}|"
            f"R{int(num_removed)}|K{int(candidates_per_removed)}"
        )

    @staticmethod
    def _coarse_profile_key(*, family: str, layers: int) -> str:
        return f"{_family_key(family)}|L{int(layers)}"

    def _build_profile_medians(self, rows: Sequence[Mapping[str, object]]) -> None:
        grouped: Dict[str, List[Mapping[str, object]]] = {}
        coarse: Dict[str, List[Mapping[str, object]]] = {}
        for row in rows:
            family = _family_key(str(row.get("family", "")))
            layers = int(_safe_float(row.get("layers"), 1))
            shots = int(_safe_float(row.get("shots"), 256))
            num_removed = int(_safe_float(row.get("num_removed"), 1))
            candidates = int(_safe_float(row.get("candidates_per_removed"), 2))
            grouped.setdefault(
                self._profile_key(
                    family=family,
                    layers=layers,
                    shots=shots,
                    num_removed=num_removed,
                    candidates_per_removed=candidates,
                ),
                [],
            ).append(row)
            coarse.setdefault(self._coarse_profile_key(family=family, layers=layers), []).append(row)

        def median_profile(items: Sequence[Mapping[str, object]]) -> Dict[str, float]:
            return {
                "circ_depth": float(np.median([_safe_float(item.get("circ_depth"), 1.0) for item in items])),
                "circ_twoq_ops": float(np.median([_safe_float(item.get("circ_twoq_ops"), 0.0) for item in items])),
                "num_vars": float(np.median([_safe_float(item.get("num_vars"), 1.0) for item in items])),
            }

        self.profile_medians = {key: median_profile(items) for key, items in grouped.items()}
        self.profile_medians.update({f"coarse:{key}": median_profile(items) for key, items in coarse.items()})

        if rows:
            self.default_profile = median_profile(rows)

    def _feature_dict_from_context(self, context: CircuitNoiseContext) -> Dict[str, float]:
        family = _family_key(context.family)
        num_vars = context.num_vars
        if num_vars is None:
            num_vars = int(context.num_removed) * int(context.candidates_per_removed)

        key = self._profile_key(
            family=family,
            layers=context.layers,
            shots=context.shots,
            num_removed=context.num_removed,
            candidates_per_removed=context.candidates_per_removed,
        )
        coarse_key = f"coarse:{self._coarse_profile_key(family=family, layers=context.layers)}"
        med = self.profile_medians.get(key, self.profile_medians.get(coarse_key, self.default_profile))

        circ_depth = _safe_float(context.circ_depth, med.get("circ_depth", 1.0))
        circ_twoq_ops = _safe_float(context.circ_twoq_ops, med.get("circ_twoq_ops", 0.0))

        return {
            "bias": 1.0,
            "is_qaoa": 1.0 if family == "qaoa" else 0.0,
            "is_vqa_su2": 1.0 if family == "vqa_su2" else 0.0,
            "layers": float(context.layers),
            "log_shots": math.log1p(max(1, int(context.shots))),
            "num_removed": float(context.num_removed),
            "candidates_per_removed": float(context.candidates_per_removed),
            "num_vars": float(num_vars),
            "circ_depth": float(circ_depth),
            "circ_twoq_ops": float(circ_twoq_ops),
            "optimization_level": float(context.optimization_level),
        }

    def _feature_dict_from_row(self, row: Mapping[str, object]) -> Dict[str, float]:
        return self._feature_dict_from_context(
            CircuitNoiseContext(
                family=str(row.get("family", "")),
                layers=int(_safe_float(row.get("layers"), 1)),
                shots=int(_safe_float(row.get("shots"), 256)),
                num_removed=int(_safe_float(row.get("num_removed"), 1)),
                candidates_per_removed=int(_safe_float(row.get("candidates_per_removed"), 2)),
                num_vars=int(_safe_float(row.get("num_vars"), 1)),
                circ_depth=_safe_float(row.get("circ_depth"), 1.0),
                circ_twoq_ops=_safe_float(row.get("circ_twoq_ops"), 0.0),
                optimization_level=int(_safe_float(row.get("optimization_level"), 3)),
            )
        )

    def _matrix(self, feature_dicts: Sequence[Mapping[str, float]]) -> np.ndarray:
        return np.asarray([[float(item.get(name, 0.0)) for name in self.feature_names] for item in feature_dicts], dtype=float)

    def _standardize_fit(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.where(std <= 1e-12, 1.0, std)
        self.feature_mean = mean.tolist()
        self.feature_std = std.tolist()
        return (x - mean) / std

    def _standardize_predict(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.feature_mean, dtype=float)
        std = np.asarray(self.feature_std, dtype=float)
        std = np.where(std <= 1e-12, 1.0, std)
        return (x - mean) / std

    def fit(self, rows: Sequence[Mapping[str, object]]) -> None:
        if not rows:
            raise ValueError("Cannot fit hardware-noise model on an empty dataset.")

        self.rows_seen = len(rows)
        self.backend_names = sorted({str(row.get("backend_name", "")) for row in rows if row.get("backend_name")})
        self._build_profile_medians(rows)

        feature_dicts = [self._feature_dict_from_row(row) for row in rows]
        x_raw = self._matrix(feature_dicts)
        x_std = self._standardize_fit(x_raw)

        self.coefficients = {}
        target_columns: Dict[str, np.ndarray] = {}
        for row in rows:
            targets = self._row_targets(row)
            for name, value in targets.items():
                target_columns.setdefault(name, []).append(value)  # type: ignore[arg-type]

        x_design = x_std
        penalty = self.ridge_alpha * np.eye(x_design.shape[1])
        penalty[0, 0] = 0.0
        for target_name, target_values in target_columns.items():
            y = np.asarray(target_values, dtype=float)
            coef = np.linalg.pinv(x_design.T @ x_design + penalty) @ (x_design.T @ y)
            self.coefficients[target_name] = coef.tolist()
            self.target_scales[target_name] = _percentile_scale(y)

    def predict(self, context: CircuitNoiseContext) -> HardwareNoisePrediction:
        if not self.coefficients:
            raise ValueError("Hardware-noise model has not been fitted or loaded.")

        feature_dict = self._feature_dict_from_context(context)
        x_raw = self._matrix([feature_dict])
        x_std = self._standardize_predict(x_raw)

        raw: Dict[str, float] = {}
        for target_name, coef_values in self.coefficients.items():
            coef = np.asarray(coef_values, dtype=float)
            raw[target_name] = float((x_std @ coef)[0])

        tvd = _clip01(raw.get("predicted_tvd", 0.0))
        entropy_gap = max(0.0, raw.get("predicted_entropy_gap", 0.0))
        feasible_loss = _clip01(raw.get("predicted_feasible_loss", 0.0))
        optimum_loss = _clip01(raw.get("predicted_optimum_loss", 0.0))
        runtime_sec = max(0.0, raw.get("predicted_runtime_sec", 0.0))

        normalized = {
            "predicted_tvd": tvd,
            "predicted_entropy_gap": _clip01(entropy_gap / self.target_scales.get("predicted_entropy_gap", 1.0)),
            "predicted_feasible_loss": feasible_loss,
            "predicted_optimum_loss": optimum_loss,
            "predicted_runtime_sec": _clip01(runtime_sec / self.target_scales.get("predicted_runtime_sec", 1.0)),
        }
        noise_score = 0.0
        weight_total = 0.0
        for name, weight in self.target_weights.items():
            noise_score += float(weight) * normalized.get(name, 0.0)
            weight_total += float(weight)
        if weight_total > 0:
            noise_score /= weight_total

        return HardwareNoisePrediction(
            predicted_tvd=tvd,
            predicted_entropy_gap=entropy_gap,
            predicted_feasible_loss=feasible_loss,
            predicted_optimum_loss=optimum_loss,
            predicted_runtime_sec=runtime_sec,
            noise_score=_clip01(noise_score),
        )

    def predict_min_noise_score(
        self,
        *,
        num_removed: int,
        candidates_per_removed: int,
        shots: int,
        families: Iterable[str] = ("qaoa", "vqa_su2"),
        layers: Iterable[int] = (1, 2),
        optimization_level: int = 3,
    ) -> float:
        scores = []
        for family in families:
            for layer in layers:
                prediction = self.predict(
                    CircuitNoiseContext(
                        family=family,
                        layers=int(layer),
                        shots=int(shots),
                        num_removed=int(num_removed),
                        candidates_per_removed=int(candidates_per_removed),
                        optimization_level=int(optimization_level),
                    )
                )
                scores.append(prediction.noise_score)
        return float(min(scores)) if scores else 1.0

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "model_type": "empirical_hardware_noise_ridge",
            "feature_names": self.feature_names,
            "ridge_alpha": self.ridge_alpha,
            "target_weights": self.target_weights,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "coefficients": self.coefficients,
            "target_scales": self.target_scales,
            "profile_medians": self.profile_medians,
            "default_profile": self.default_profile,
            "rows_seen": self.rows_seen,
            "backend_names": self.backend_names,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "EmpiricalHardwareNoiseModel":
        model = cls(
            feature_names=list(payload.get("feature_names", NOISE_FEATURE_NAMES)),
            ridge_alpha=float(payload.get("ridge_alpha", 1.0)),
            target_weights=dict(payload.get("target_weights", {})),
        )
        model.feature_mean = [float(v) for v in payload.get("feature_mean", [])]
        model.feature_std = [float(v) for v in payload.get("feature_std", [])]
        model.coefficients = {str(k): [float(x) for x in v] for k, v in dict(payload.get("coefficients", {})).items()}
        model.target_scales = {str(k): float(v) for k, v in dict(payload.get("target_scales", {})).items()}
        model.profile_medians = {
            str(k): {str(kk): float(vv) for kk, vv in dict(v).items()}
            for k, v in dict(payload.get("profile_medians", {})).items()
        }
        model.default_profile = {str(k): float(v) for k, v in dict(payload.get("default_profile", {})).items()}
        model.rows_seen = int(payload.get("rows_seen", 0))
        model.backend_names = [str(v) for v in payload.get("backend_names", [])]
        return model

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EmpiricalHardwareNoiseModel":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train_model_from_csv(*, hardware_csv: Path, model_path: Path, ridge_alpha: float = 1.0) -> EmpiricalHardwareNoiseModel:
    rows = EmpiricalHardwareNoiseModel.load_csv_rows(hardware_csv)
    model = EmpiricalHardwareNoiseModel(ridge_alpha=ridge_alpha)
    model.fit(rows)
    model.save(model_path)
    return model


def summarize_hardware_csv(path: Path) -> Dict[str, object]:
    rows = EmpiricalHardwareNoiseModel.load_csv_rows(path)
    if not rows:
        return {"rows": 0}
    tvd = np.asarray([_safe_float(row.get("tvd_hw_vs_aer"), 0.0) for row in rows], dtype=float)
    families = sorted({_family_key(str(row.get("family", ""))) for row in rows})
    return {
        "rows": len(rows),
        "backends": sorted({str(row.get("backend_name", "")) for row in rows if row.get("backend_name")}),
        "families": families,
        "num_removed": sorted({int(_safe_float(row.get("num_removed"), 0)) for row in rows}),
        "candidates_per_removed": sorted({int(_safe_float(row.get("candidates_per_removed"), 0)) for row in rows}),
        "shots": sorted({int(_safe_float(row.get("shots"), 0)) for row in rows}),
        "tvd_mean": float(np.mean(tvd)),
        "tvd_std": float(np.std(tvd)),
        "tvd_min": float(np.min(tvd)),
        "tvd_max": float(np.max(tvd)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or inspect an empirical hardware-noise model.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a hardware-noise model from real hardware CSV rows.")
    train.add_argument("--hardware-csv", type=Path, required=True)
    train.add_argument("--model-path", type=Path, default=Path("runs/entropy_pdptw/hardware_noise_model.json"))
    train.add_argument("--ridge-alpha", type=float, default=1.0)

    summarize = sub.add_parser("summarize", help="Summarize a hardware CSV.")
    summarize.add_argument("--hardware-csv", type=Path, required=True)

    predict = sub.add_parser("predict", help="Predict noise for one circuit context.")
    predict.add_argument("--model-path", type=Path, required=True)
    predict.add_argument("--family", type=str, default="qaoa")
    predict.add_argument("--layers", type=int, default=1)
    predict.add_argument("--shots", type=int, default=256)
    predict.add_argument("--num-removed", type=int, default=3)
    predict.add_argument("--candidate-cap", type=int, default=3)

    args = parser.parse_args()
    if args.command == "train":
        model = train_model_from_csv(
            hardware_csv=args.hardware_csv,
            model_path=args.model_path,
            ridge_alpha=args.ridge_alpha,
        )
        print(json.dumps({"model_path": str(args.model_path), "rows_seen": model.rows_seen, "backends": model.backend_names}, indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize_hardware_csv(args.hardware_csv), indent=2))
    elif args.command == "predict":
        model = EmpiricalHardwareNoiseModel.load(args.model_path)
        prediction = model.predict(
            CircuitNoiseContext(
                family=args.family,
                layers=args.layers,
                shots=args.shots,
                num_removed=args.num_removed,
                candidates_per_removed=args.candidate_cap,
            )
        )
        print(json.dumps(prediction.as_dict(), indent=2))


if __name__ == "__main__":
    main()
