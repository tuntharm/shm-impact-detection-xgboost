# Update the function to include grid lines similar to MATLAB
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle


def plot_cylinder_predictions(y_test, y_pred_theta, y_pred_z, r=11.55, z_max=45):
    """
    Plots a 3D cylinder with a grid and visualizes true vs predicted positions.

    Parameters:
        y_test (DataFrame): True impact locations with columns ["theta", "z"].
        y_pred_theta (array): Predicted theta values (radians).
        y_pred_z (array): Predicted Z coordinates.
        r (float): Radius of the cylinder.
        z_max (float): Maximum height of the cylinder.
    """
    # Generate cylinder surface for visualization
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, z_max, 100)
    Theta, Z_cylindrical = np.meshgrid(theta, z)
    X_cylindrical = r * np.cos(Theta)
    Y_cylindrical = r * np.sin(Theta)

    # Define sensor positions
    sensor_positions = np.array([
        [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0],  # At z=0
        [r, 0, z_max], [-r, 0, z_max], [0, r, z_max], [0, -r, z_max]  # At z=z_max
    ])

    # Function to draw squares
    def draw_square(ax, x, y, z, size):
        """Draws a square on the given axis."""
        half_size = size / 2
        square_x = [x - half_size, x + half_size, x + half_size, x - half_size, x - half_size]
        square_y = [y, y, y, y, y]
        square_z = [z - half_size, z - half_size, z + half_size, z + half_size, z - half_size]
        ax.plot(square_x, square_y, square_z, color="black", linewidth=2)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cylinder surface
    ax.plot_surface(X_cylindrical, Y_cylindrical, Z_cylindrical, color='grey', alpha=0.3, edgecolor='none')

    # **Grid Lines: Divide the Cylinder**
    # Divide Z direction into 5 sections (6 division points)
    z_divisions = np.linspace(0, z_max, 7)
    theta_grid = np.linspace(0, 2 * np.pi, 100)  # Smooth curves

    for z_val in z_divisions:
        X_grid = r * np.cos(theta_grid)
        Y_grid = r * np.sin(theta_grid)
        Z_grid = np.full_like(theta_grid, z_val)
        ax.plot(X_grid, Y_grid, Z_grid, 'k-', linewidth=0.5)  # Black horizontal grid lines

    # Divide Circumference into 8 sections
    theta_divisions = np.linspace(0, 2 * np.pi, 9)  # 8 sections → 9 division points
    z_grid = np.linspace(0, z_max, 100)  # Full height

    for theta_val in theta_divisions:
        X_grid = r * np.cos(theta_val) * np.ones_like(z_grid)
        Y_grid = r * np.sin(theta_val) * np.ones_like(z_grid)
        Z_grid = z_grid
        ax.plot(X_grid, Y_grid, Z_grid, 'k-', linewidth=0.5)  # Black vertical grid lines

    # Plot black squares at specified locations
    for x, y, z in sensor_positions:
        draw_square(ax, x, y, z, size=2)

    # **Plot true positions**
    true_x = np.cos(y_test["theta"]) * r
    true_y = np.sin(y_test["theta"]) * r
    true_z = y_test["z"]

    ax.scatter(true_x, true_y, true_z, color="blue", label="True Position", s=40)

    # **Plot predicted positions**
    pred_x = np.cos(y_pred_theta) * r
    pred_y = np.sin(y_pred_theta) * r
    pred_z = y_pred_z
    ax.scatter(pred_x, pred_y, pred_z, color="red", label="Predicted Position", s=40)

    # **Draw grey dashed lines connecting true and predicted positions**
    for tx, ty, tz, px, py, pz in zip(true_x, true_y, true_z, pred_x, pred_y, pred_z):
        ax.plot([tx, px], [ty, py], [tz, pz], linestyle="dashed", color="grey", alpha=0.6)

    # Function to set equal axis ratio
    def set_axes_equal(ax):
        """Make the 3D plot axes have equal scale."""
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range) / 2

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    set_axes_equal(ax)

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("True vs Predicted Positions on Cylinder with Grid")
    ax.legend()

    # Show plot
    plt.show(block=False)






import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_tankflatten_predictions(
    y_test, pred_theta, y_pred_z, test_loc,
    rmse_total=None, rmse_z=None, rmse_theta=None, accuracy=None,
    r=11.55, z_max=45, FP_mask=None):
    """
    Plot true vs predicted impact positions on a flattened cylindrical tank.

    Parameters:
        y_test (DataFrame): Ground truth with columns "sin_theta", "cos_theta", "z"
        pred_theta (ndarray): Predicted theta values (radians)
        y_pred_z (ndarray): Predicted Z values (mm)
        test_loc (str): "top" or "full" — affects theta wrapping logic
        rmse_total (float): Total RMSE in cm
        rmse_z (float): Z RMSE in cm
        rmse_theta (float): Theta RMSE in cm
        accuracy (float): Accuracy score
        r (float): Radius of cylinder (mm)
        z_max (float): Max Z height (mm)
        FP_mask (array-like): Boolean mask for false positives
    """
    circumference = 2 * np.pi * r

    true_theta = np.arctan2(y_test["sin_theta"], y_test["cos_theta"])

    # Flattened coordinates
    true_x_flat = y_test["z"].to_numpy()
    true_y_flat = (r * true_theta).to_numpy()
    pred_x_flat = y_pred_z
    pred_y_flat = r * pred_theta

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.scatter(true_x_flat, true_y_flat, s=100, c='blue', marker='+', label='True Position', linewidths=2)
    plt.scatter(pred_x_flat, pred_y_flat, s=100, c='red', marker='+', label='Predicted Position', linewidths=2)

    for i in range(len(y_test)):
        plt.plot([true_x_flat[i], pred_x_flat[i]], [true_y_flat[i], pred_y_flat[i]], 'k--', linewidth=1.2)

    # Unwrapped cylinder rectangle
    x_rect = [0, z_max, z_max, 0, 0]
    y_rect = [circumference/2, circumference/2, -circumference/2, -circumference/2, circumference/2]
    plt.plot(x_rect, y_rect, 'k-', linewidth=2)

    # Grid lines
    for i in range(1, 8):
        y_h = circumference/2 - i * (circumference / 8)
        plt.plot([0, z_max], [y_h, y_h], 'k-', linewidth=0.5)

    for j in range(1, 6):
        x_v = j * (z_max / 6)
        plt.plot([x_v, x_v], [-circumference/2, circumference/2], 'k-', linewidth=0.5)

    # Sensor locations
    sensor_theta = np.linspace(-np.pi, np.pi, 5)
    sensor_positions = np.vstack([
        np.column_stack([np.zeros(5), -r * sensor_theta]),
        np.column_stack([np.ones(5) * z_max, -r * sensor_theta])
    ])
    S_labels = ['S3', 'S4', 'S1', 'S2', 'S3', 'S7', 'S8', 'S5', 'S6', 'S7']
    for i, label in enumerate(S_labels):
        plt.text(sensor_positions[i, 0], sensor_positions[i, 1], label,
                 color='blue', fontsize=12, fontweight='bold', ha='center')

    # False positive error circles
    if FP_mask is not None:
        for i in np.where(FP_mask)[0]:
            cx = true_x_flat[i]
            cy = true_y_flat[i]
            dx = cx - pred_x_flat[i]
            dy = cy - pred_y_flat[i]
            err_r = np.sqrt(dx**2 + dy**2)
            circle = Circle((cx, cy), err_r, color='red', fill=False, linestyle='--', linewidth=1.2, alpha=0.5)
            plt.gca().add_patch(circle)

    # Title and layout
    title_text = r'$\bf{Flattened\ Cylinder:\ True\ vs\ Predicted\ Positions}$'
    if rmse_total is not None:
        rmse_str = f'RMSE: {rmse_total:.3f} cm'
        if rmse_theta is not None and rmse_z is not None:
            rmse_str += f' (Theta: {rmse_theta:.3f}, Z: {rmse_z:.3f})'
        title_text += f"\n{rmse_str} - Accuracy: {accuracy:.3f}"

    plt.title(title_text, fontsize=14)
    plt.xlabel('Z-axis (Height) (cm)')
    plt.ylabel('Unwrapped Circumference Position (cm)')
    plt.xlim([-10, z_max + 10])
    plt.ylim([-circumference/2 - 10, circumference/2 + 10])
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
def plot_plate_predictions(y_test, y_pred_x, y_pred_y):
    ## XGB--------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test["Loc_X"], y_test["Loc_Y"], color="blue", label="True Position")
    plt.scatter(y_pred_x, y_pred_y, color="red", label="Predicted Position (XGBoost)")

    # Draw grey dashed lines connecting true and predicted positions
    for true_x, true_y, pred_x, pred_y in zip(y_test["Loc_X"], y_test["Loc_Y"], y_pred_x, y_pred_y):
        plt.plot([true_x, pred_x], [true_y, pred_y], linestyle="dashed", color="grey", alpha=0.6)

    plt.xlabel("Loc_X")
    plt.ylabel("Loc_Y")
    plt.title("True vs Predicted Positions (XGBoost)")
    plt.xlim((-80,80))
    plt.ylim((-60,60))
    plt.legend()
    plt.grid()
    plt.show(block=False)
    