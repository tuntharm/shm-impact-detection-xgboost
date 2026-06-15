from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .geometry import add_angular_targets, add_sensor_geometry

BASE_FEATURE_PREFIXES = ("ToA_", "Amplitude_", "SignalEnergy_", "PeakFreq_")
LOCALISATION_EXTRA_PREFIXES = ("S",)
TARGET_COLUMNS = {"Loc", "theta", "z", "sin_theta", "cos_theta", "Impact_Type"}


def direct_csv_files(data_dirs: Iterable[str | Path]) -> list[Path]:
    """Return CSV files directly inside each directory, without recursive scanning."""
    files: list[Path] = []
    for raw_dir in data_dirs:
        directory = Path(raw_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Data directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory}")
        files.extend(sorted(path for path in directory.glob("*.csv") if path.is_file()))
    if not files:
        dirs = ", ".join(str(Path(item)) for item in data_dirs)
        raise FileNotFoundError(f"No direct CSV files found in: {dirs}")
    return files


def load_feature_data(data_dirs: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate feature CSVs from one or more explicit directories."""
    files = direct_csv_files(data_dirs)
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_feature_data(df: pd.DataFrame, z_max: float) -> pd.DataFrame:
    """Append geometry and trigonometric target features used by the models."""
    return add_angular_targets(add_sensor_geometry(df, z_max=z_max))


def localisation_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns for impact localisation, excluding labels and targets."""
    columns: list[str] = []
    for column in df.columns:
        if column in TARGET_COLUMNS or column == "source_file":
            continue
        if column == "Force_N":
            columns.append(column)
            continue
        if column.startswith(BASE_FEATURE_PREFIXES):
            columns.append(column)
            continue
        if column.startswith("S") and (column.endswith("_theta") or column.endswith("_z")):
            columns.append(column)
    if not columns:
        raise ValueError("No localisation feature columns were found.")
    return columns


def classification_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns for hard/soft impact classification."""
    if "Impact_Type" not in df.columns:
        raise ValueError("Missing Impact_Type target column for classification.")

    columns: list[str] = []
    for column in df.columns:
        if column == "Force_N" or column.startswith(BASE_FEATURE_PREFIXES):
            columns.append(column)
    if not columns:
        raise ValueError("No classification feature columns were found.")
    return columns
