import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from plot_utils import plot_plate_predictions
from savefile import save_predictions
import os
import glob
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import time





# data = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/Features_stlham_p1_tank.csv"
# df = pd.read_csv(data)

folder_path = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/tank/top"

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
ambiguous_locs = [8, 18, 28, 38]  # Loc column indicating ambiguous hits
if "Loc" in df.columns:
    df = df[~df["Loc"].isin(ambiguous_locs)].reset_index(drop=True)
    print(f"Data filtered. Remaining rows: {len(df)}")
#--------------TRAINING--------------------------
#------------------------------------------------

# ------------------ FEATURE & TARGET SETUP ------------------
feature_columns = df.loc[:, "ToA_S1":"Force_N"].columns
X = df[feature_columns]
y = df[["theta", "z"]]

# ------------------ SPLIT DATA ------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.05, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ------------------ PARAMETER GRID ------------------
param_grid = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.015, 0.02, 0.025, 0.03],
    'max_depth': [6, 8, 10],
    'subsample': [0.4, 0.6, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0.6, 0.7, 0.8],
    'reg_lambda': [2.5, 3.0, 3.5]
}

# ------------------ FUNCTION TO TUNE A TARGET ------------------
def tune_xgboost(target_label):
    print(f"\n--- Tuning for: {target_label} ---")
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        scoring='neg_root_mean_squared_error',
        n_iter=20,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    start = time.time()
    search.fit(X_train, y_train[target_label])
    end = time.time()

    best_params = search.best_params_
    best_rmse = -search.best_score_

    print(f"Best Parameters for {target_label}: {best_params}")
    print(f"Best RMSE for {target_label}: {best_rmse:.4f}")
    print(f"Tuning Time: {end - start:.2f} seconds")

    return search.best_estimator_, best_rmse, best_params

# ------------------ RUN TUNING FOR Z & THETA ------------------
best_model_z, best_rmse_z, best_params_z = tune_xgboost("z")
best_model_theta, best_rmse_theta, best_params_theta = tune_xgboost("theta")

# ------------------ FINAL EVALUATION ------------------
y_pred_theta = best_model_theta.predict(X_test)
y_pred_z = best_model_z.predict(X_test)

rmse_theta_final = np.sqrt(mean_squared_error(y_test["theta"], y_pred_theta))
rmse_z_final = np.sqrt(mean_squared_error(y_test["z"], y_pred_z))
rmse_total_final = np.sqrt(
    np.mean(
        (y_test["theta"] - y_pred_theta) ** 2 +
        (y_test["z"] - y_pred_z) ** 2
    )
)

print("\n====== FINAL TEST RMSE ======")
print(f"Test RMSE (theta): {rmse_theta_final:.4f}")
print(f"Test RMSE (z): {rmse_z_final:.4f}")
print(f"Test RMSE (total): {rmse_total_final:.4f}")