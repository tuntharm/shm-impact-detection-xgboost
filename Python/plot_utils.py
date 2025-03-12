# Update the function to include grid lines similar to MATLAB
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_cylinder_predictions(y_test, y_pred_x, y_pred_y, y_pred_z, r=11.55, z_max=45):
    """
    Plots a 3D cylinder with a grid and visualizes true vs predicted positions.

    Parameters:
        y_test (DataFrame): True impact locations with columns ["Loc_X", "Loc_Y", "Loc_Z"].
        y_pred_x (array): Predicted X coordinates.
        y_pred_y (array): Predicted Y coordinates.
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
        [11.5, 0, 0], [-11.5, 0, 0], [0, 11.5, 0], [0, -11.5, 0],  # At z=0
        [11.5, 0, z_max], [-11.5, 0, z_max], [0, 11.5, z_max], [0, -11.5, z_max]  # At z=z_max
    ])

    # Function to draw squares
    def draw_square(ax, x, y, z, size):
        """Draws a square on the given axis."""
        half_size = size / 2
        square_x = [x - half_size, x + half_size, x + half_size, x - half_size, x - half_size]
        square_z = [z - half_size, z - half_size, z + half_size, z + half_size, z - half_size]
        square_y = [y, y, y, y, y]
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

    # Plot true positions
    ax.scatter(y_test["Loc_X"], y_test["Loc_Y"], y_test["Loc_Z"], color="blue", label="True Position", s=40)

    # Plot predicted positions
    ax.scatter(y_pred_x, y_pred_y, y_pred_z, color="red", label="Predicted Position", s=40)

    # Draw grey dashed lines connecting true and predicted positions
    for tx, ty, tz, px, py, pz in zip(y_test["Loc_X"], y_test["Loc_Y"], y_test["Loc_Z"], y_pred_x, y_pred_y, y_pred_z):
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
    ax.set_xlabel("Loc_X")
    ax.set_ylabel("Loc_Y")
    ax.set_zlabel("Loc_Z")
    ax.set_title("True vs Predicted Positions on Cylinder with Grid")
    ax.legend()

    # Show plot
    plt.show()




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_flat_predictions(y_test, y_pred_x, y_pred_y, r=11.55, z_max=45):
    """
    Plots a flattened cylindrical representation (2D) with grid and visualizes true vs predicted positions.

    Parameters:
        y_test (DataFrame): True impact locations with columns ["Loc_X", "Loc_Y"].
        y_pred_x (array): Predicted X coordinates (axial).
        y_pred_y (array): Predicted Y coordinates (circumferential, unrolled).
        r (float): Radius of the cylinder.
        z_max (float): Maximum width of the flattened cylinder.
    """
    # Define unwrapped height
    height = 2 * np.pi * r  # Total height of the unwrapped cylinder

    # Extract true values
    x_true = y_test["Loc_X"].to_numpy()  # Axial positions
    y_true = y_test["Loc_Y"].to_numpy()  # Circumferential positions (already unrolled)

    # Grid lines
    num_h_lines = 7  # Horizontal grid divisions (Circumferential)
    num_v_lines = 5  # Vertical grid divisions (Axial)

    # Generate grid line positions
    y_grid_lines = np.linspace(-height/2, height/2, num_h_lines + 2)  # Horizontal grid lines
    x_grid_lines = np.linspace(0, z_max, num_v_lines + 2)  # Vertical grid lines

    # Create 2D plot
    plt.figure(figsize=(10, 6))

    # Plot horizontal grid lines
    for y_h in y_grid_lines:
        plt.plot([0, z_max], [y_h, y_h], 'k-', linewidth=0.5)  # Black lines

    # Plot vertical grid lines
    for x_v in x_grid_lines:
        plt.plot([x_v, x_v], [-height/2, height/2], 'k-', linewidth=0.5)  # Black lines

    # Define sensor positions (mapping to 2D)
    sensor_positions = np.array([
        [0, 0],                          # S1
        [0, -2 * (2 * np.pi * r / 8)],   # S2
        [0, -4 * (2 * np.pi * r / 8)],   # S3
        [0, -6 * (2 * np.pi * r / 8)],   # S4
        [-z_max, 0],                     # S5
        [-z_max, -2 * (2 * np.pi * r / 8)],  # S6
        [-z_max, -4 * (2 * np.pi * r / 8)],  # S7
        [-z_max, -6 * (2 * np.pi * r / 8)]   # S8
    ])

    # Plot black squares at sensor positions
    for x_s, y_s in sensor_positions:
        plt.scatter(x_s, y_s, color="black", s=100, marker="s")

    # Plot true positions (blue dots)
    plt.scatter(x_true, y_true, color="blue", label="True Position", s=40)

    # Plot predicted positions (red crosses)
    plt.scatter(y_pred_x, y_pred_y, color="red", marker='x', label="Predicted Position", s=40)

    # Draw grey dashed lines connecting true and predicted positions
    for tx, ty, px, py in zip(x_true, y_true, y_pred_x, y_pred_y):
        plt.plot([tx, px], [ty, py], linestyle="dashed", color="grey", alpha=0.6)

    # Formatting
    plt.xlabel("Axial Position (X)")
    plt.ylabel("Circumferential Position (Unwrapped Y)")
    plt.title("True vs Predicted Positions (Flattened Cylinder)")
    plt.xlim(-z_max, 0)
    plt.ylim(-height,0 )
    plt.legend()
    plt.grid(False)  # Default grid is disabled since we created custom lines
    plt.show()
