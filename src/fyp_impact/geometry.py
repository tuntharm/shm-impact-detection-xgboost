from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_RADIUS_CM = 11.55
DEFAULT_Z_MAX_CM = 45.0
DEFAULT_THRESHOLD_CM = 3.5
DEFAULT_TOP_LOCS = tuple(list(range(1, 16)) + list(range(31, 41)))
AMBIGUOUS_LOCS = (8, 18, 28, 38)

SENSOR_THETA = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2) * 2
SENSOR_Z = (0.0, 0.0, 0.0, 0.0, DEFAULT_Z_MAX_CM, DEFAULT_Z_MAX_CM, DEFAULT_Z_MAX_CM, DEFAULT_Z_MAX_CM)


def add_sensor_geometry(df: pd.DataFrame, z_max: float = DEFAULT_Z_MAX_CM) -> pd.DataFrame:
    """Return a copy with fixed cylindrical sensor coordinates appended."""
    result = df.copy()
    sensor_z = (0.0, 0.0, 0.0, 0.0, z_max, z_max, z_max, z_max)
    for index, (theta, z_pos) in enumerate(zip(SENSOR_THETA, sensor_z), start=1):
        result[f"S{index}_theta"] = theta
        result[f"S{index}_z"] = z_pos
    return result


def add_angular_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with sin/cos encoded angular targets."""
    required = {"theta", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing localisation target columns: {sorted(missing)}")

    result = df.copy()
    result["sin_theta"] = np.sin(result["theta"])
    result["cos_theta"] = np.cos(result["theta"])
    return result


def angular_difference(pred_theta: np.ndarray, true_theta: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference, wrapped to [-pi, pi]."""
    return (np.asarray(pred_theta) - np.asarray(true_theta) + np.pi) % (2 * np.pi) - np.pi


def reconstructed_theta(pred_sin: np.ndarray, pred_cos: np.ndarray) -> np.ndarray:
    """Recover angular position from independently predicted sin/cos targets."""
    return np.arctan2(pred_sin, pred_cos)


def parse_location_set(value: str | Iterable[int] | None) -> tuple[int, ...]:
    """Parse comma-separated locations and ranges such as '1-15,31-40'."""
    if value is None:
        return DEFAULT_TOP_LOCS
    if not isinstance(value, str):
        return tuple(int(item) for item in value)

    locations: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            locations.extend(range(int(start), int(end) + 1))
        else:
            locations.append(int(part))
    return tuple(dict.fromkeys(locations))


def top_location_mask(locations: pd.Series, top_locs: Iterable[int] = DEFAULT_TOP_LOCS) -> pd.Series:
    """Boolean mask for the top-half impact locations used in the report."""
    return locations.isin(tuple(top_locs))
