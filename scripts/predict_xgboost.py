from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fyp_impact.data import (  # noqa: E402
    classification_feature_columns,
    load_feature_data,
    localisation_feature_columns,
    prepare_feature_data,
)
from fyp_impact.geometry import DEFAULT_RADIUS_CM, DEFAULT_TOP_LOCS, DEFAULT_Z_MAX_CM, parse_location_set, top_location_mask  # noqa: E402
from fyp_impact.metrics import spatial_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from saved XGBoost model JSON files.")
    parser.add_argument("--data-dir", action="append", required=True, help="Directory containing direct feature CSV files.")
    parser.add_argument("--model-dir", default="outputs/xgboost/models", help="Directory containing model JSON files from train_xgboost.py.")
    parser.add_argument("--output-csv", default="outputs/xgboost/inference_predictions.csv", help="CSV path for predictions.")
    parser.add_argument("--test-loc", choices=("top", "full"), default="full", help="Optionally filter to report top-half locations.")
    parser.add_argument("--top-locs", default="1-15,31-40", help="Location set used when --test-loc top.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_CM, help="Tank radius in cm for optional error columns.")
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX_CM, help="Tank axial length in cm.")
    return parser.parse_args()


def load_regressor(path: Path) -> xgb.XGBRegressor:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def load_classifier(path: Path) -> xgb.XGBClassifier | None:
    if not path.exists():
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_feature_data(load_feature_data(args.data_dir), z_max=args.z_max)
    if args.test_loc == "top":
        top_locs = parse_location_set(args.top_locs)
        df = df.loc[top_location_mask(df["Loc"], top_locs)].reset_index(drop=True)
    else:
        top_locs = DEFAULT_TOP_LOCS

    sin_model = load_regressor(model_dir / "localisation_sin_theta.json")
    cos_model = load_regressor(model_dir / "localisation_cos_theta.json")
    z_model = load_regressor(model_dir / "localisation_z.json")

    loc_features = localisation_feature_columns(df)
    pred_sin = sin_model.predict(df[loc_features])
    pred_cos = cos_model.predict(df[loc_features])
    pred_theta = np.arctan2(pred_sin, pred_cos)
    pred_z = z_model.predict(df[loc_features])

    output = pd.DataFrame(
        {
            "source_file": df["source_file"],
            "Loc": df["Loc"],
            "pred_theta": pred_theta,
            "pred_z": pred_z,
        }
    )

    if {"theta", "z"} <= set(df.columns):
        output["true_theta"] = df["theta"]
        output["true_z"] = df["z"]
        output["spatial_error_cm"] = spatial_errors(df["theta"], pred_theta, df["z"], pred_z, radius=args.radius)

    classifier = load_classifier(model_dir / "impact_type_classifier.json")
    if classifier is not None and "Impact_Type" in df.columns:
        cls_features = classification_feature_columns(df)
        output["pred_impact_type"] = classifier.predict(df[cls_features])
        output["true_impact_type"] = df["Impact_Type"]

    output.to_csv(output_csv, index=False)
    print(f"Saved {len(output)} predictions to {output_csv}")
    if args.test_loc == "top":
        print(f"Filtered to top locations: {','.join(str(item) for item in top_locs)}")


if __name__ == "__main__":
    main()
