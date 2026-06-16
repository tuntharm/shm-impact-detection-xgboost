from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fyp_impact.data import load_feature_data, localisation_feature_columns, prepare_feature_data  # noqa: E402
from fyp_impact.geometry import DEFAULT_RADIUS_CM, DEFAULT_Z_MAX_CM  # noqa: E402
from fyp_impact.metrics import spatial_errors  # noqa: E402
from fyp_impact.models import default_xgb_regressor_params, train_localisation_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample XGBoost localisation hyperparameters.")
    parser.add_argument("--data-dir", action="append", required=True, help="Directory containing direct feature CSV files. Repeat for A/B/C combinations.")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_CM)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX_CM)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.random_state)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_feature_data(load_feature_data(args.data_dir), z_max=args.z_max)
    features = localisation_feature_columns(df)
    x_values = df[features]
    y_values = df[["sin_theta", "cos_theta", "z"]]
    x_train, x_temp, y_train, y_temp = train_test_split(x_values, y_values, test_size=0.2, random_state=args.random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=args.random_state)

    grid = {
        "n_estimators": [400, 500, 600, 700],
        "learning_rate": [0.015, 0.02, 0.025, 0.03],
        "max_depth": [6, 8, 10],
        "subsample": [0.4, 0.6, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_alpha": [0.5, 0.6, 0.8],
        "reg_lambda": [3.0, 3.5, 4.0],
    }
    combinations = [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]
    sampled = random.sample(combinations, min(args.samples, len(combinations)))
    base_params = default_xgb_regressor_params()
    best: dict[str, object] | None = None
    results: list[dict[str, object]] = []

    for index, params in enumerate(sampled, start=1):
        model_params = {**base_params, **params, "random_state": args.random_state}
        models = train_localisation_models(x_train, y_train, x_val, y_val, params=model_params)
        pred_theta, pred_z = models.predict(x_val)
        true_theta = np.arctan2(y_val["sin_theta"], y_val["cos_theta"])
        mean_error = float(np.mean(spatial_errors(true_theta, pred_theta, y_val["z"], pred_z, radius=args.radius)))
        result = {"index": index, "mean_spatial_error_cm": mean_error, "params": params}
        results.append(result)
        print(f"{index:03d}/{len(sampled)} mean spatial error: {mean_error:.4f} cm")
        if best is None or mean_error < best["mean_spatial_error_cm"]:
            best = result

    if best is None:
        raise RuntimeError("No hyperparameter combinations were evaluated.")

    final_params = {**base_params, **best["params"], "random_state": args.random_state}
    final_models = train_localisation_models(x_train, y_train, x_val, y_val, params=final_params)
    pred_theta, pred_z = final_models.predict(x_test)
    true_theta = np.arctan2(y_test["sin_theta"], y_test["cos_theta"])
    test_mean_error = float(np.mean(spatial_errors(true_theta, pred_theta, y_test["z"], pred_z, radius=args.radius)))
    payload = {"best": best, "test_mean_spatial_error_cm": test_mean_error, "results": results}
    (output_dir / "tuning_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["best"], indent=2))
    print(f"Test mean spatial error: {test_mean_error:.4f} cm")


if __name__ == "__main__":
    main()
