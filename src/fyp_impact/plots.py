from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import angular_difference


def save_flattened_tank_plot(
    output_path: str | Path,
    y_test: pd.DataFrame,
    pred_theta: np.ndarray,
    pred_z: np.ndarray,
    radius: float,
    z_max: float,
    threshold_cm: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    true_theta = np.arctan2(y_test["sin_theta"], y_test["cos_theta"])
    true_x = y_test["z"].to_numpy()
    true_y = radius * true_theta.to_numpy()
    pred_x = np.asarray(pred_z)
    pred_y = radius * np.asarray(pred_theta)
    circumference = 2 * np.pi * radius
    errors = np.sqrt((angular_difference(pred_theta, true_theta) * radius) ** 2 + (pred_z - y_test["z"]) ** 2)

    plt.figure(figsize=(10, 6))
    plt.scatter(true_x, true_y, s=80, c="tab:blue", marker="+", label="True")
    plt.scatter(pred_x, pred_y, s=80, c="tab:red", marker="+", label="Predicted")
    for tx, ty, px, py in zip(true_x, true_y, pred_x, pred_y):
        plt.plot([tx, px], [ty, py], color="0.45", linestyle="--", linewidth=0.9)

    x_rect = [0, z_max, z_max, 0, 0]
    y_rect = [circumference / 2, circumference / 2, -circumference / 2, -circumference / 2, circumference / 2]
    plt.plot(x_rect, y_rect, color="black", linewidth=1.5)
    for index in range(1, 8):
        y_grid = circumference / 2 - index * circumference / 8
        plt.plot([0, z_max], [y_grid, y_grid], color="black", linewidth=0.4)
    for index in range(1, 6):
        x_grid = index * z_max / 6
        plt.plot([x_grid, x_grid], [-circumference / 2, circumference / 2], color="black", linewidth=0.4)

    for index in np.where(errors > threshold_cm)[0]:
        circle = Circle((true_x[index], true_y[index]), errors[index], fill=False, linestyle="--", color="tab:red", alpha=0.45)
        plt.gca().add_patch(circle)

    plt.title("Flattened tank impact localisation")
    plt.xlabel("Axial position z (cm)")
    plt.ylabel("Unwrapped circumference r theta (cm)")
    plt.xlim([-5, z_max + 5])
    plt.ylim([-circumference / 2 - 5, circumference / 2 + 5])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_confusion_matrix(output_path: str | Path, matrix: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Soft", "Hard"], yticklabels=["Soft", "Hard"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Impact type classification")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_feature_importance(output_path: str | Path, model, title: str) -> None:
    import matplotlib.pyplot as plt
    import xgboost as xgb

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xgb.plot_importance(model, importance_type="weight", max_num_features=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
