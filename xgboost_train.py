import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os
import glob

from plot_utils import plot_cylinder_predictions
from plot_utils import plot_tankflatten_predictions
from savefile import save_predictions



# data = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/Features_stlham_p1_tank.csv"
# df = pd.read_csv(data)

folder_path = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/tank/"

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
num_rows, num_columns = df.shape

print(f"Rows: {num_rows}, Columns: {num_columns}")

# ---------------- FILTER OUT AMBIGUOUS LOCATIONS ----------------
# ambiguous_locs = [8, 18, 28, 38]  # Loc column indicating ambiguous hits
# if "Loc" in df.columns:
#     df = df[~df["Loc"].isin(ambiguous_locs)].reset_index(drop=True)
#     print(f"Data filtered. Remaining rows: {len(df)}")

df["is_ambiguous"] = df["Loc"].isin([8, 18, 28, 38]).astype(int)


#--------------TRAINING--------------------------
#------------------------------------------------
# Selecting features (all columns from "Distance_S1" to "Force_N")
feature_columns = df.loc[:, "ToA_S1":"Force_N"].columns
X = df[feature_columns]

# Selecting target variables (impact location coordinates)
y = df[["theta", "z"]]  



# Split data:
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -------------- XGBoost PARAMETERS --------------------------
xgb_params = {
    'objective': "reg:squarederror",
    'n_estimators': 600,  # Increase trees for better learning
    'learning_rate': 0.02,  # Lower learning rate for stability
    'max_depth': 10,  # Deeper trees for better feature learning
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'reg_lambda': 3,  # Regularization to prevent overfitting
    'reg_alpha': 0.5, # Regularization to prevent overfitting
    'tree_method': 'hist',
    'device': 'cuda',}
# -------------- TRAINING XGBoost --------------------------
# Train XGBoost for Loc_X
xgb_model_theta = xgb.XGBRegressor(**xgb_params)
xgb_model_theta.fit(
    X_train, y_train["theta"],
    eval_set=[(X_val, y_val["theta"])],
    verbose=False
)

# Train XGBoost for Loc_Y
xgb_model_z = xgb.XGBRegressor(**xgb_params)
xgb_model_z.fit(
    X_train, y_train["z"],
    eval_set=[(X_val, y_val["z"])],
    verbose=False
)


# Make predictions
y_pred_theta = xgb_model_theta.predict(X_test)
y_pred_z = xgb_model_z.predict(X_test)
#--------------------------------------------------------------------------
# Compute RMSE
rmse_theta = np.sqrt(mean_squared_error(y_test["theta"], y_pred_theta))
rmse_z = np.sqrt(mean_squared_error(y_test["z"], y_pred_z))
rmse_total = np.sqrt(
    np.mean(
        (y_test["theta"] - y_pred_theta) ** 2 +
        (y_test["z"] - y_pred_z) ** 2
    )
)
print("--------Training complete--------")

# Store RMSE as an array [X, Y, Z, Total]
rmse = np.array([rmse_theta, rmse_z, rmse_total])
print(f"RMSE for Loc_theta: {rmse_theta:.4f}")
print(f"RMSE for Loc_Z: {rmse_z:.4f}")
print(f"RMSE : {rmse_total:.4f}")


#plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model_theta, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances for Loc_theta")
#plt.show()

#plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model_z, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances for Loc_Z")
#plt.show()


#------------------------------------------------
r = 11.55
z_max = 45
theta_error = np.abs(y_test["theta"] - y_pred_theta)
z_error = np.abs(y_test["z"] - y_pred_z)

# Define thresholds
error_distance = 2.5 # DEFINE ***********************************
error_ang = error_distance/r

TP_mask = (theta_error <= error_ang) & (z_error <= error_distance)
FP_mask = ~TP_mask  # All others are incorrect

true_positives = TP_mask.sum()
false_positives = FP_mask.sum()

print(f"TP: {true_positives}, FP: {false_positives}")

# Compute Accuracy
accuracy = true_positives / len(y_test)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / len(y_test)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")



#------------------------------------------------

## VISUALISATION
plot_cylinder_predictions(y_test, y_pred_theta, y_pred_z)
plot_tankflatten_predictions(y_test, y_pred_theta, y_pred_z, rmse_total, rmse_theta, rmse_z, r=11.55, z_max=45, FP_mask=FP_mask) 
#plot_plate_predictions(y_test, y_pred_theta, y_pred_z) #For plate
filename = 'predictions_XGB.mat'
save_predictions(y_test, y_pred_theta, y_pred_z, filename, rmse)

