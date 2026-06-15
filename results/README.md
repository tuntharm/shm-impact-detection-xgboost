# Results Evidence Pack

This folder records the public-facing result story without adding new claims beyond the final report.

## Authoritative Source

The committed report at [`../docs/report/FYP_02048996.pdf`](../docs/report/FYP_02048996.pdf) is the source of the metrics below. The public repository includes a compact processed CSV sample for workflow reproduction, while the larger A/B/C data folders and Colab experiments remain in access-controlled Google Drive folders.

## Localisation, 10-Fold Cross-Validation

| Dataset setting | Spatial RMSE | Accuracy within 3.5 cm |
| --- | ---: | ---: |
| A+B | 1.94 +/- 0.24 cm | 84.98 +/- 3.53% |
| A+B+C | 1.35 +/- 0.17 cm | 91.39 +/- 2.78% |
| B+C Full | 1.39 +/- 0.23 cm | 90.85 +/- 2.28% |
| B+C Top | 1.07 +/- 0.23 cm | 93.86 +/- 2.25% |

## Hard/Soft Impact Classification

The report's B+C Top classification table reports:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Soft | 100.0% | 99.26% | 99.6% | 135 |
| Hard | 99.2% | 100.0% | 99.6% | 128 |
| Overall accuracy |  |  | 99.6% | 263 |

The 10-fold classification accuracy summary reports:

| Dataset setting | Accuracy |
| --- | ---: |
| A+B | 99.77 +/- 0.36% |
| A+B+C | 99.47 +/- 0.55% |
| B+C Full | 99.39 +/- 0.86% |
| B+C Top | 99.36 +/- 1.00% |

## Public Sample Outputs

Running:

```bash
python scripts/train_xgboost.py --data-dir data/processed/tank/16april --test-loc top --output-dir outputs/xgboost
```

creates a local evidence bundle:

- `outputs/xgboost/metrics.json`
- `outputs/xgboost/predictions_localisation.csv`
- `outputs/xgboost/predictions_classification.csv`
- `outputs/xgboost/flattened_tank_predictions.png`
- `outputs/xgboost/confusion_matrix.png`
- `outputs/xgboost/importance_*.png`
- `outputs/xgboost/models/*.json`
- `outputs/xgboost/inference_predictions.csv` after running `scripts/predict_xgboost.py`

Those generated outputs are ignored by Git because they are reproducible artifacts rather than source files.

Grouped validation can be run as an additional stress test:

```bash
python scripts/grouped_validate.py --data-dir data/processed/tank/16april --group-by Loc --folds 5 --test-loc top
```

This is included as validation tooling, not as an extra report result claim.

## Interpretation

The report positions XGBoost as the strongest overall public-facing model because it combines low localisation error, strong hard/soft classification, and much lower runtime than the neural baselines. ConvXGB slightly improves classification accuracy in one comparison table, but it costs substantially more runtime and does not improve localisation error.
