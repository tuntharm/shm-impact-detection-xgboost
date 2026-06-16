# Notebook and Drive Provenance

The project used Google Colab notebooks and Drive-hosted data during development. The raw notebooks are not committed here because they contain bulky outputs, generated plots, and environment-specific paths. The public repository instead keeps cleaned scripts and documentation.

## Access-Controlled Drive Folders

- Main project / Colab folder: <https://drive.google.com/drive/folders/1eV1nWm934i87P8r-wzCiRNaIlF0wvnsz>
- A/B/C processed data folder: <https://drive.google.com/drive/folders/1XZoWibuwQfjbE8UywY4qdUQTbJVWTcx2>
- `16april` processed CSV folder: <https://drive.google.com/drive/folders/1fUreyAiRBNO5NepapWqen2SD_-f8QQnH>

These folders may require owner-granted access.

## Referenced Colab Notebooks

- `main_XGBtrain.ipynb`: <https://colab.research.google.com/drive/1Xd44GR7gjjo9EV3DRD9GtuzAgg8PXUty>
- `main_ConvXGBtrain.ipynb`: <https://colab.research.google.com/drive/1XMeru_X4vyV56MFSewB4WZdJEPNcG7cL>
- `unitcircle_XGBtrain.ipynb`: <https://colab.research.google.com/drive/1Rxi9MhHp9W4_14hnJDhh1dj4H2ndVi_O>
- `unitcircle_ANNtrain.ipynb`: <https://colab.research.google.com/drive/1HBLRAsPXaChXX9bRE8JD5UXbOlMaZgLz>

## Why Scripts Are Preferred Publicly

The cleaned `scripts/` entrypoints are easier to review than notebooks because they avoid hidden state, embedded output blobs, and accidental private paths. They also make the public workflow easier to test in CI.
