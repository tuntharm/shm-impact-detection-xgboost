from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED_MODELS = (
    "localisation_sin_theta.json",
    "localisation_cos_theta.json",
    "localisation_z.json",
    "impact_type_classifier.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated public workflow artifacts without fixing metric values.")
    parser.add_argument("--metrics", required=True, help="metrics.json from train_xgboost.py.")
    parser.add_argument("--predictions", required=True, help="Inference prediction CSV from predict_xgboost.py.")
    parser.add_argument("--grouped-metrics", required=True, help="grouped_validation_metrics.json from grouped_validate.py.")
    parser.add_argument("--model-dir", required=True, help="Directory containing saved XGBoost JSON models.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def require_finite(value: object, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Expected finite numeric value for {label}, got {value!r}")


def verify_predictions(path: str | Path) -> None:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {csv_path}")
    required = {"source_file", "Loc", "pred_theta", "pred_z"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")


def verify_models(path: str | Path) -> None:
    model_dir = Path(path)
    for name in EXPECTED_MODELS:
        model_path = model_dir / name
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty model file: {model_path}")


def main() -> None:
    args = parse_args()
    metrics = load_json(args.metrics)
    grouped = load_json(args.grouped_metrics)

    if metrics.get("rows", 0) <= 0:
        raise ValueError("Training metrics must report a positive row count.")
    require_finite(metrics["localisation"]["rmse_total_cm"], "localisation.rmse_total_cm")
    require_finite(metrics["localisation"]["accuracy_within_threshold"], "localisation.accuracy_within_threshold")
    require_finite(metrics["classification"]["report"]["accuracy"], "classification.report.accuracy")

    if grouped.get("folds_completed", 0) <= 0:
        raise ValueError("Grouped validation must complete at least one fold.")
    require_finite(grouped["localisation_rmse_total_cm"]["mean"], "grouped.localisation_rmse_total_cm.mean")
    require_finite(grouped["classification_accuracy"]["mean"], "grouped.classification_accuracy.mean")

    verify_predictions(args.predictions)
    verify_models(args.model_dir)
    print("Public workflow artifacts verified.")


if __name__ == "__main__":
    main()
