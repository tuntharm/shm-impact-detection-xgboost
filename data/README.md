# Data

This repository includes a compact processed-data sample at:

`data/processed/tank/16april`

The loader intentionally reads only CSV files directly inside each `--data-dir`. This makes data combinations explicit:

```bash
python scripts/train_xgboost.py --data-dir data/processed/tank/16april
python scripts/train_xgboost.py --data-dir data/processed/tank/16april/A
```

The full experiment data, raw signals, and cloud notebook assets are not committed here. They remain in Google Drive and may require access from the owner:

- Main project/Colab folder: https://drive.google.com/drive/folders/1eV1nWm934i87P8r-wzCiRNaIlF0wvnsz
- A/B/C processed data folder: https://drive.google.com/drive/folders/1XZoWibuwQfjbE8UywY4qdUQTbJVWTcx2
- `16april` processed CSV folder: https://drive.google.com/drive/folders/1fUreyAiRBNO5NepapWqen2SD_-f8QQnH
