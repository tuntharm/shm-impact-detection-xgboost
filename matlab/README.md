# MATLAB Signal Processing

The MATLAB folder preserves the raw-signal-to-feature stage from the project. It is intentionally documented as a preprocessing layer, while the main public training interface lives in Python.

- [`tank/`](tank/) contains the cylindrical tank feature extraction used by the public Python workflow.
- [`plate/`](plate/) contains the earlier flat-panel feature extraction scripts.

The main generators are configured with portable relative defaults and optional environment variables. Update `FYP_TANK_RAW_DIR`, `FYP_TANK_PROCESSED_DIR`, `FYP_PLATE_RAW_DIR`, or `FYP_PLATE_PROCESSED_DIR` before running them against private raw data. The stable interface between MATLAB and Python is the processed CSV schema documented in [`../docs/methodology/feature-extraction.md`](../docs/methodology/feature-extraction.md).
