# XGBoost Methodology

XGBoost is the primary public workflow because it gives strong performance from compact, auditable features and avoids the heavier training footprint of the neural baselines.

## Inputs

The feature table contains:

- `Loc`, `theta`, `z` target coordinates,
- per-sensor `ToA_S*`,
- per-sensor `Amplitude_S*`,
- per-sensor `SignalEnergy_S*`,
- `Force_N`,
- `Impact_Type` for hard/soft classification.

The Python loader appends fixed sensor coordinates for the eight PZT sensors: four around the lower axial end and four around the upper axial end.

## Localisation Target Design

Directly regressing `theta` is fragile because cylindrical angles wrap at `-pi` and `pi`. The model therefore trains three independent regressors:

- `sin(theta)`,
- `cos(theta)`,
- `z`.

Prediction reconstructs the angle with `atan2(pred_sin, pred_cos)`. Error is then computed on the unwrapped cylindrical surface:

```text
theta_error_cm = shortest_angular_difference(pred_theta, true_theta) * radius
z_error_cm = pred_z - true_z
spatial_error_cm = sqrt(theta_error_cm^2 + z_error_cm^2)
```

The default radius is `11.55 cm`, the axial length is `45 cm`, and the report threshold is `3.5 cm`.

## Default Hyperparameters

The public scripts use the final-report XGBoost settings by default:

| Parameter | Search range in report | Public default |
| --- | --- | ---: |
| `n_estimators` | 400 to 1000 | 600 |
| `learning_rate` | 0.01 to 0.20 | 0.020 |
| `max_depth` | 3 to 12 | 10 |
| `subsample` | 0.4 to 1.0 | 0.6 |
| `colsample_bytree` | 0.5 to 1.0 | 0.8 |
| `reg_lambda` | 2.0 to 5.0 | 3.0 |
| `reg_alpha` | 0.5 to 1.0 | 0.5 |

The scripts use `tree_method="hist"` so the default run is CPU-safe. GPU acceleration was useful in cloud experiments but is not required for the public workflow.

## Classification

The classifier is an XGBoost binary classifier trained on waveform-derived features and `Force_N`. It predicts hard versus soft impact class using `Impact_Type` as the target. The public script writes a classification report, confusion matrix, feature-importance plot, and prediction table.

## Validation Protocols

The report-facing public cross-validation script uses shuffled K-fold validation over processed rows. Because related samples can share impact locations, cases, or heads, the repository also includes `scripts/grouped_validate.py` for stricter stress testing:

```bash
python scripts/grouped_validate.py --data-dir data/processed/tank/16april --group-by Loc --folds 5 --test-loc top
```

Grouped validation results are not added to the headline report tables here. They are included as tooling for future work and external due diligence.
