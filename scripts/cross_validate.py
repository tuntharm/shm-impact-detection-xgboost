from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-fold validation for the XGBoost workflow.")
    parser.add_argument("--data-dir", action="append", required=True, help="Directory containing direct feature CSV files. Repeat for A/B/C combinations.")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--test-loc", choices=("top", "full"), default="top")
    parser.add_argument("--top-locs", default="1-15,31-40")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_CM)
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX_CM)
    parser.add_argument("--threshold-cm", type=float, default=DEFAULT_THRESHOLD_CM)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/cross_validation")
    return parser.parse_args()


def summarise(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_locs = parse_location_set(args.top_locs) if args.test_loc == "top" else DEFAULT_TOP_LOCS

    df = prepare_feature_data(load_feature_data(args.data_dir), z_max=args.z_max)
    loc_features = localisation_feature_columns(df)
    cls_features = classification_feature_columns(df)
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    regressor_params = default_xgb_regressor_params(n_estimators=args.n_estimators)
    classifier_params = default_xgb_classifier_params(n_estimators=args.n_estimators)

    fold_results: list[dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(kfold.split(df), start=1):
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        train_inner, val_inner = train_test_split(train_df, test_size=0.1, random_state=args.random_state + fold_index)

        if args.test_loc == "top":
            mask = top_location_mask(test_df["Loc"], top_locs)
            test_df = test_df.loc[mask].reset_index(drop=True)
        if test_df.empty:
            continue

        loc_models = train_localisation_models(
            train_inner[loc_features],
            train_inner[["sin_theta", "cos_theta", "z"]],
            val_inner[loc_features],
            val_inner[["sin_theta", "cos_theta", "z"]],
            params=regressor_params,
        )
        pred_theta, pred_z = loc_models.predict(test_df[loc_features])
        true_theta = np.arctan2(test_df["sin_theta"], test_df["cos_theta"])
        loc_metrics = regression_metrics(true_theta, pred_theta, test_df["z"], pred_z, args.radius, args.threshold_cm)

        classifier = train_impact_classifier(train_df[cls_features], train_df["Impact_Type"], params=classifier_params)
        pred_cls = classifier.predict(test_df[cls_features])
        class_accuracy = float(accuracy_score(test_df["Impact_Type"], pred_cls))
        fold_results.append(
            {
                "fold": fold_index,
                "localisation": loc_metrics,
                "classification_accuracy": class_accuracy,
            }
        )
        print(
            f"Fold {fold_index:02d}: RMSE={loc_metrics['rmse_total_cm']:.3f} cm, "
            f"threshold_acc={loc_metrics['accuracy_within_threshold']:.3f}, "
            f"class_acc={class_accuracy:.3f}"
        )

    summary = {
        "rows": int(len(df)),
        "folds_requested": args.folds,
        "folds_completed": len(fold_results),
        "test_loc": args.test_loc,
        "localisation_rmse_total_cm": summarise([item["localisation"]["rmse_total_cm"] for item in fold_results]),
        "localisation_accuracy_within_threshold": summarise(
            [item["localisation"]["accuracy_within_threshold"] for item in fold_results]
        ),
        "classification_accuracy": summarise([item["classification_accuracy"] for item in fold_results]),
        "fold_results": fold_results,
    }
    (output_dir / "cross_validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
