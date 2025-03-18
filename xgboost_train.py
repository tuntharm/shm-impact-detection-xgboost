import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from plot_utils import plot_cylinder_predictions
from plot_utils import plot_plate_predictions
from savefile import save_predictions
import os
import glob





# data = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/Features_stlham_p1_tank.csv"
# df = pd.read_csv(data)

folder_path = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/tank"

# Get all CSV files in the folder (ignoring subfolders)
csv_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if os.path.isfile(f)]

# Check if files exist
if not csv_files:
    raise FileNotFoundError("No CSV files found in the specified folder.")

# Initialize an empty list to store DataFrames
dfs = []

# Load each CSV file
for file in csv_files:
    data = pd.read_csv(file)  # Read CSV
    dfs.append(data)  # Append to list

# Concatenate all dataframes
df = pd.concat(dfs, ignore_index=True)

# Display the first few rows of the combined DataFrame
#print(df.head())
num_rows, num_columns = df.shape

print(f"Rows: {num_rows}, Columns: {num_columns}")




#--------------TRAINING--------------------------
#------------------------------------------------
# Selecting features (all columns from "Distance_S1" to "Force_N")
feature_columns = df.loc[:, "ToA_S1":"Force_N"].columns
X = df[feature_columns]

# Selecting target variables (impact location coordinates)
y = df[["theta", "z"]]  # Multi-output regression targets

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.05, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -------------- XGBoost PARAMETERS --------------------------
xgb_params = {
    'objective': "reg:squarederror",
    'n_estimators': 500,  # Increase trees for better learning
    'learning_rate': 0.02,  # Lower learning rate for stability
    'max_depth': 9,  # Deeper trees for better feature learning
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'reg_lambda': 3,  # Regularization to prevent overfitting
    'reg_alpha': 0.5 # Regularization to prevent overfitting
}

# -------------- TRAINING XGBoost --------------------------
# Train XGBoost for Loc_X
xgb_model_x = xgb.XGBRegressor(**xgb_params)
xgb_model_x.fit(
    X_train, y_train["theta"],
    eval_set=[(X_val, y_val["z"])],
    verbose=False
)

# Train XGBoost for Loc_Y
xgb_model_y = xgb.XGBRegressor(**xgb_params)
xgb_model_y.fit(
    X_train, y_train["z"],
    eval_set=[(X_val, y_val["z"])],
    verbose=False
)

# Make predictions
y_pred_x_xgb = xgb_model_x.predict(X_test)
y_pred_y_xgb = xgb_model_y.predict(X_test)

# Compute RMSE
rmse_x_xgb = np.sqrt(mean_squared_error(y_test["theta"], y_pred_x_xgb))
rmse_y_xgb = np.sqrt(mean_squared_error(y_test["z"], y_pred_y_xgb))
rmse_total_xgb = np.sqrt(
    np.mean(
        (y_test["theta"] - y_pred_x_xgb) ** 2 +
        (y_test["z"] - y_pred_y_xgb) ** 2 
    )
)
print("Training complete.")

#------------------------------------------------
#------------------------------------------------
# Store RMSE as an array [X, Y, Z, Total]
rmse_xgb = np.array([rmse_x_xgb, rmse_y_xgb, rmse_total_xgb])
print(f"RMSE for Loc_theta: {rmse_x_xgb:.4f}")
print(f"RMSE for Loc_Z: {rmse_y_xgb:.4f}")
print(f"RMSE : {rmse_total_xgb:.4f}")

#plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model_x, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances for Loc_theta")
#plt.show()

#plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model_y, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances for Loc_Z")
#plt.show()


#------------------------------------------------

## VISUALISATION
plot_cylinder_predictions(y_test, y_pred_x_xgb, y_pred_y_xgb)
#plot_tankflatten_predictions(y_test, y_pred_x_xgb, y_pred_y_xgb)
#plot_plate_predictions(y_test, y_pred_x_xgb, y_pred_y_xgb) #For plate
filename = 'predictions_XGB.mat'
save_predictions(y_test, y_pred_x_xgb, y_pred_y_xgb, filename, rmse_xgb)

