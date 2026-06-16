from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .geometry import DEFAULT_RADIUS_CM, angular_difference


def spatial_errors(
    true_theta: np.ndarray | pd.Series,
    pred_theta: np.ndarray | pd.Series,
    true_z: np.ndarray | pd.Series,
    pred_z: np.ndarray | pd.Series,
    radius: float = DEFAULT_RADIUS_CM,
) -> np.ndarray:
    """Euclidean impact localisation error on the unwrapped cylindrical surface."""
    theta_error = angular_difference(np.asarray(pred_theta), np.asarray(true_theta)) * radius
    z_error = np.asarray(pred_z) - np.asarray(true_z)
    return np.sqrt(theta_error**2 + z_error**2)


def regression_metrics(
    true_theta: np.ndarray | pd.Series,
    pred_theta: np.ndarray | pd.Series,
    true_z: np.ndarray | pd.Series,
    pred_z: np.ndarray | pd.Series,
    radius: float,
    threshold_cm: float,
) -> dict[str, float | int]:
    """Return report-style localisation metrics."""
    theta_error = angular_difference(np.asarray(pred_theta), np.asarray(true_theta))
    errors = spatial_errors(true_theta, pred_theta, true_z, pred_z, radius=radius)
    positives = errors <= threshold_cm
    rmse_theta = float(np.sqrt(np.mean((theta_error * radius) ** 2)))
    rmse_z = float(np.sqrt(mean_squared_error(true_z, pred_z)))
    rmse_total = float(np.sqrt(np.mean(errors**2)))
    return {
        "samples": int(len(errors)),
        "rmse_theta_cm": rmse_theta,
        "rmse_z_cm": rmse_z,
        "rmse_total_cm": rmse_total,
        "threshold_cm": float(threshold_cm),
        "accuracy_within_threshold": float(np.mean(positives)) if len(errors) else 0.0,
        "true_positives": int(positives.sum()),
        "false_positives": int((~positives).sum()),
    }
