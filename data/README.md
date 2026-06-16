# Data

This repository includes compact processed tank feature CSVs at:

```text
data/processed/tank/16april
```

The files are processed feature tables, not raw voltage traces. Raw experiment folders, full A/B/C data, and cloud notebook outputs remain in access-controlled Google Drive.

## Loader Rule

The Python loader reads only CSV files directly inside each `--data-dir`. It does not recurse into child folders. This is deliberate: A/B/C combinations should be explicit and reviewable.

```bash
python scripts/train_xgboost.py --data-dir data/processed/tank/16april
python scripts/train_xgboost.py --data-dir data/processed/tank/16april/A
python scripts/train_xgboost.py --data-dir data/processed/tank/16april --data-dir data/processed/tank/16april/A
```

## Public Sample Inventory

Direct `16april` CSV files:

| File | Rows | Bytes | SHA256 |
| --- | ---: | ---: | --- |
| `Features_rubham_p10_tank_new.csv` | 160 | 78,563 | `19CFDB445F61A137A0CB22E08A9D2547103E161C3B9BB077D3989DD603FBD1D5` |
| `Features_rubham_p10_tank_top.csv` | 250 | 122,241 | `B9C8C45CD2EFB72E8380BFA345D434FF7115A8BC091759C70E50F114B5A52C0F` |
| `Features_silham_p1_tank_new.csv` | 160 | 77,806 | `0950DC22B49D2F479243BC7AE846C7B2C1892EFF3AE7FAD7F76EEFF360812A0D` |
| `Features_silham_p1_tank_top.csv` | 250 | 121,749 | `4AD6C9135A840418ED68EC893180BB53F747D2DCC29FDA5FB47A077990C02E06` |
| `Features_srubham_p10_tank_new.csv` | 160 | 78,648 | `B4DF0085BE90300A2C04EADF7B7D75C8D235C8B85624C16243D6763B4ECBEC47` |
| `Features_srubham_p10_tank_top.csv` | 250 | 122,486 | `5D9266BE6A4072043D46444F6A5DDD2D2A1AC6A2008D68D7A42CA2C766EED5A7` |
| `Features_stlham_p1_tank_new.csv` | 160 | 77,846 | `5ACB046D9EE367A2763C7A97BE5FF5F25E7808F6690B860E3CBD639C4D8615B5` |
| `Features_stlham_p1_tank_top.csv` | 250 | 121,796 | `A0B6FD5908A179018271EE8A46F310796F4A5FAF5821D0E636B42310943BD7EA` |

Nested `16april/A` CSV files:

| File | Rows | Bytes | SHA256 |
| --- | ---: | ---: | --- |
| `A/Features_rubham_p10_tank.csv` | 159 | 78,142 | `256C131CB958A62AB51854B2FA9875D02EE1172763BCCFBCB9CB6E3D05335B63` |
| `A/Features_silham_p1_tank.csv` | 160 | 77,918 | `8A4019B931E15C0EA6AFC980538216F25D425C9DF0A31B7B56C8DDFE6EC49262` |
| `A/Features_srubham_p10_tank.csv` | 158 | 77,655 | `9554C9BD25193E828CB49AD4F7101CCC15278867980DD5304BACCB329C13DFCE` |
| `A/Features_stlham_p1_tank.csv` | 161 | 78,029 | `28156A0BE0C6FAE3F592A7A51ACD8F958CF40850F7F7EB93D9D243AE556D006A` |

## CSV Schema

| Column family | Meaning |
| --- | --- |
| `Loc` | Discrete impact location ID. |
| `theta` | Tank angular coordinate in radians. |
| `z` | Tank axial coordinate in centimetres. |
| `ToA_S1` to `ToA_S8` | Per-sensor time-of-arrival features after event-level normalisation. |
| `Amplitude_S1` to `Amplitude_S8` | Per-sensor dominant signed amplitude after filtering and normalisation. |
| `SignalEnergy_S1` to `SignalEnergy_S8` | Per-sensor signal energy feature. |
| `Force_N` | Peak impact force in Newtons from the force-channel conversion. |
| `Impact_Type` | Binary hard/soft label used by the classifier. |

The feature extraction lineage is documented in [`../docs/methodology/feature-extraction.md`](../docs/methodology/feature-extraction.md).

## Full Data Provenance

Drive folders used during the project:

- Main project / Colab folder: <https://drive.google.com/drive/folders/1eV1nWm934i87P8r-wzCiRNaIlF0wvnsz>
- A/B/C processed data folder: <https://drive.google.com/drive/folders/1XZoWibuwQfjbE8UywY4qdUQTbJVWTcx2>
- `16april` processed CSV folder: <https://drive.google.com/drive/folders/1fUreyAiRBNO5NepapWqen2SD_-f8QQnH>

These links may require owner-granted access.
