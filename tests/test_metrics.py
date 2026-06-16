from __future__ import annotations

import numpy as np

from fyp_impact.geometry import angular_difference
from fyp_impact.metrics import regression_metrics, spatial_errors


def test_angular_difference_wraps_at_pi_boundary():
    pred = np.array([-np.pi + 0.1])
    true = np.array([np.pi - 0.1])

    diff = angular_difference(pred, true)

    assert np.allclose(diff, np.array([0.2]))


def test_spatial_errors_use_shortest_angular_path():
    pred = np.array([-np.pi + 0.1])
    true = np.array([np.pi - 0.1])

    errors = spatial_errors(true, pred, np.array([10.0]), np.array([10.0]), radius=10.0)

    assert np.allclose(errors, np.array([2.0]))


def test_regression_metrics_threshold_accuracy():
    true_theta = np.array([0.0, 0.0])
    pred_theta = np.array([0.1, 1.0])
    true_z = np.array([0.0, 0.0])
    pred_z = np.array([0.0, 0.0])

    metrics = regression_metrics(true_theta, pred_theta, true_z, pred_z, radius=10.0, threshold_cm=3.5)

    assert metrics["samples"] == 2
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["accuracy_within_threshold"] == 0.5
