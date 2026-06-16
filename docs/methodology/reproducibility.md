# Reproducibility

This repository separates three levels of reproducibility.

## Level 1: Public Code Path

The public sample data is sufficient to verify that the loader, geometry transforms, XGBoost training, metrics, and plots run end to end:

```bash
python scripts/train_xgboost.py --data-dir data/processed/tank/16april --test-loc top --output-dir outputs/xgboost
python scripts/cross_validate.py --data-dir data/processed/tank/16april --folds 10 --test-loc top
python scripts/predict_xgboost.py --data-dir data/processed/tank/16april --model-dir outputs/xgboost/models --output-csv outputs/xgboost/inference_predictions.csv --test-loc top
```

Expected generated files are listed in [`../../results/README.md`](../../results/README.md). These outputs are intentionally ignored by Git.

## Level 2: Report Result Traceability

The full final-report metrics are traceable to [`../report/FYP_02048996.pdf`](../report/FYP_02048996.pdf). The committed report provides the official result tables, model comparison, and discussion.

The public sample is compact and may not match report numbers exactly because it does not contain every raw recording, processed A/B/C split, or cloud notebook output used during the project.

## Level 3: Full Experiment Reproduction

Full experiment reproduction requires the access-controlled Google Drive folders:

- main project / Colab folder,
- A/B/C processed data folder,
- `16april` processed CSV folder,
- cloud notebooks used for XGBoost, ConvXGB, and ANN experiments.

Those folders are documented in [`../../notebooks/README.md`](../../notebooks/README.md) and [`../../data/README.md`](../../data/README.md). The README does not promise anonymous access because sharing state is controlled outside the Git repository.

## Environment

Minimum local workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall src scripts
pytest
```

Known Colab notebook environment observed in project notebooks:

| Package | Version |
| --- | --- |
| Python | 3.11.12 |
| XGBoost | 2.1.4 |
| NumPy | 2.0.2 |
| Pandas | 2.2.2 |
| scikit-learn | 1.6.1 |

## Validation Protocol Caveat

The public scripts include both row-shuffled train/test or K-fold workflows and an optional grouped validation workflow:

```bash
python scripts/grouped_validate.py --data-dir data/processed/tank/16april --group-by Loc --folds 5 --test-loc top
```

The grouped script is a due-diligence tool for stronger external research claims. Its outputs are not promoted as additional final-report results.
