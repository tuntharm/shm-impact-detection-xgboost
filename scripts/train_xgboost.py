from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fyp_impact.data import (  # noqa: E402
    classification_feature_columns,
    load_feature_data,
    localisation_feature_columns,
    prepare_feature_data,
)
from fyp_impact.geometry import (  # noqa: E402
    DEFAULT_RADIUS_CM,
    DEFAULT_THRESHOLD_CM,
    DEFAULT_TOP_LOCS,
    DEFAULT_Z_MAX_CM,
    parse_location_set,
    top_location_mask,
)
from fyp_impact.metrics import regression_metrics  # noqa: E402
from fyp_impact.models import (  # noqa: E402
    default_xgb_classifier_params,
    default_xgb_regressor_params,
    train_impact_classifier,
    train_localisation_models,
)
from fyp_impact.plots import save_confusion_matrix, save_feature_importance, save_flattened_tank_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost impact localisation and classification models.")
    parser.add_argument("--data-dir", action="append", required=True, help="Directory containing direct feature CSV files. Repeat for A/B/C combinations.")
    parser.add_argument("--output-dir", default="outputs/xgboost", help="Directory for metrics and plots.")
    parser.add_argument("--test-loc", choices=("top", "full"), default="top", help="Evaluate localisation on all locations or top-half locations only.")
    parser.add_argument("--top-locs", default="1-15,31-40", help="Location set used when --test-loc top.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_CM, help="Tank radius in cm.")
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX_CM, help="Tank axial length in cm.")
    parser.add_argument("--threshold-cm", type=float, default=DEFAULT_THRESHOLD_CM, help="Spatial error threshold for accuracy.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Number of XGBoost trees.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits and models.")
    parser.add_argument("--skip-classification", action="store_true", help="Only train localisation regressors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_locs = parse_location_set(args.top_locs) if args.test_loc == "top" else DEFAULT_TOP_LOCS

    raw_df = load_feature_data(args.data_dir)
    df = prepare_feature_data(raw_df, z_max=args.z_max)
    loc_features = localisation_feature_columns(df)
    x_values = df[loc_features]
    y_values = df[["sin_theta", "cos_theta", "z"]]

    x_train, x_temp, y_train, y_temp = train_test_split(x_values, y_values, test_size=0.1, random_state=args.random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=args.random_state)
    test_locations = df.loc[y_test.index, "Loc"].reset_index(drop=True)

    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    if args.test_loc == "top":
        mask = top_location_mask(test_locations, top_locs).reset_index(drop=True)
        x_test = x_test.loc[mask].reset_index(drop=True)
        y_test = y_test.loc[mask].reset_index(drop=True)
        test_locations = test_locations.loc[mask].reset_index(drop=True)

    regressor_params = default_xgb_regressor_params(n_estimators=args.n_estimators)
    regressor_params["random_state"] = args.random_state
    loc_models = train_localisation_models(x_train, y_train, x_val, y_val, params=regressor_params)
    pred_theta, pred_z = loc_models.predict(x_test)
    true_theta = np.arctan2(y_test["sin_theta"], y_test["cos_theta"])
    localisation_metrics = regression_metrics(
        true_theta=true_theta,
        pred_theta=pred_theta,
        true_z=y_test["z"],
        pred_z=pred_z,
        radius=args.radius,
        threshold_cm=args.threshold_cm,
    )

    save_flattened_tank_plot(
        output_dir / "flattened_tank_predictions.png",
        y_test=y_test,
        pred_theta=pred_theta,
        pred_z=pred_z,
        radius=args.radius,
        z_max=args.z_max,
        threshold_cm=args.threshold_cm,
    )
    save_feature_importance(output_dir / "importance_sin_theta.png", loc_models.sin_model, "Top features for sin(theta)")
    save_feature_importance(output_dir / "importance_cos_theta.png", loc_models.cos_model, "Top features for cos(theta)")
    save_feature_importance(output_dir / "importance_z.png", loc_models.z_model, "Top features for z")

    metrics: dict[str, object] = {
        "rows": int(len(df)),
        "data_dirs": args.data_dir,
        "test_loc": args.test_loc,
        "test_locations": [int(item) for item in test_locations.tolist()],
        "localisation_features": loc_features,
        "localisation": localisation_metrics,
        "regressor_params": regressor_params,
    }

    if not args.skip_classification:
        cls_features = classification_feature_columns(df)
        x_cls = df[cls_features]
        y_cls = df["Impact_Type"]
        x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
            x_cls, y_cls, test_size=0.2, random_state=args.random_state, stratify=y_cls
        )
        classifier_params = default_xgb_classifier_params(n_estimators=args.n_estimators)
        classifier_params["random_state"] = args.random_state
        classifier = train_impact_classifier(x_train_cls, y_train_cls, params=classifier_params)
        y_pred_cls = classifier.predict(x_test_cls)
        report = classification_report(y_test_cls, y_pred_cls, digits=4, output_dict=True)
        matrix = confusion_matrix(y_test_cls, y_pred_cls)
        save_confusion_matrix(output_dir / "confusion_matrix.png", matrix)
        save_feature_importance(output_dir / "importance_classification.png", classifier, "Top features for impact type")
        metrics["classification_features"] = cls_features
        metrics["classification"] = {
            "report": report,
            "confusion_matrix": matrix.tolist(),
            "classifier_params": classifier_params,
        }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
