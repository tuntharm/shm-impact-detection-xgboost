# Model Comparison

This folder summarises the report-backed comparison between the three modelling families used in the project. It does not introduce new result claims beyond the final report.

## Models

| Model | Role in project | Strength | Tradeoff |
| --- | --- | --- | --- |
| XGBoost | Primary public workflow | Fast, interpretable feature importances, strongest localisation result | Depends on engineered waveform features |
| ConvXGB | Hybrid neural/boosted baseline | Slightly higher hard/soft classification accuracy in the report table | Slower and weaker localisation than XGBoost |
| ANN | Dense neural baseline | Useful sanity-check baseline | Higher localisation error and slower training/runtime |

## Report-Backed B+C Top Comparison

### Localisation

| Model | RMSE | 3.5 cm threshold accuracy | Runtime |
| --- | ---: | ---: | ---: |
| XGBoost | 1.07 +/- 0.23 cm | 93.86 +/- 2.25% | 58.06 s |
| ConvXGB | 2.01 +/- 0.39 cm | 91.00 +/- 3.59% | 187.99 s |
| ANN | 3.83 +/- 0.39 cm | 77.60 +/- 4.96% | 159.30 s |

### Classification

| Model | Accuracy | Runtime |
| --- | ---: | ---: |
| XGBoost | 99.36 +/- 1.00% | 4.89 s |
| ConvXGB | 99.63 +/- 0.55% | 49.57 s |
| ANN | 96.04 +/- 1.92% | 80.13 s |

## Why XGBoost Is the Main Interface

XGBoost is the default repo path because it is the best balance of accuracy, inspectability, runtime, and reproducibility for the public processed-feature sample. It also maps cleanly to the SHM workflow: signal features can be audited sensor-by-sensor, model feature importance can be inspected, and the output can be plotted on a flattened tank coordinate system.

## What Would Strengthen the Comparison

The final report recommends broader future work rather than claiming deployment readiness:

- more impact locations and higher-density spatial sampling,
- larger sensor networks,
- pressure and boundary-condition variation,
- expanded impact classification beyond hard/soft,
- CNN and raw-signal learning baselines,
- broader hyperparameter optimisation.
