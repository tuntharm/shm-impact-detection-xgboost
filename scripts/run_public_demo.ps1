param(
    [int]$Estimators = 100,
    [int]$Folds = 3
)

$ErrorActionPreference = "Stop"

python -m compileall src scripts
pytest
python scripts/train_xgboost.py --data-dir data/processed/tank/16april --test-loc top --output-dir outputs/xgboost --n-estimators $Estimators
python scripts/predict_xgboost.py --data-dir data/processed/tank/16april --model-dir outputs/xgboost/models --output-csv outputs/xgboost/inference_predictions.csv --test-loc top
python scripts/cross_validate.py --data-dir data/processed/tank/16april --folds $Folds --test-loc top --output-dir outputs/cross_validation --n-estimators $Estimators
python scripts/grouped_validate.py --data-dir data/processed/tank/16april --group-by Loc --folds $Folds --test-loc top --output-dir outputs/grouped_validation --n-estimators $Estimators
python scripts/verify_public_outputs.py --metrics outputs/xgboost/metrics.json --predictions outputs/xgboost/inference_predictions.csv --grouped-metrics outputs/grouped_validation/grouped_validation_metrics.json --model-dir outputs/xgboost/models

Write-Host "Public demo complete. See outputs/xgboost, outputs/cross_validation, and outputs/grouped_validation."
