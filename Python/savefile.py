import scipy.io as sio
import numpy as np

def save_predictions(y_test, y_pred_x, y_pred_y, filename, rmse):
    """
    Save test and predicted values along with RMSE array to a .mat file.

    Parameters:
        y_test (DataFrame): True location values
        y_pred_x, y_pred_y, y_pred_z (array): Predicted locations
        filename (str): Output .mat filename
        rmse (array): RMSE values [rmse_x, rmse_y, rmse_z, rmse_total]
    """

    data = {
        'y_test': np.array(y_test),  
        'y_pred_x': np.array(y_pred_x),
        'y_pred_y': np.array(y_pred_y),
        'rmse': rmse  # RMSE stored as an array
    }

    sio.savemat(filename, data)
    print(f"Data saved to {filename}")
