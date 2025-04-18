import xgboost as xgb
import numpy as np
import pandas as pd
import os, glob, time
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random

# --------------------------------------
# CONFIG
# --------------------------------------
r = 11.55  # radius of cylinder
NUM_RANDOM_SAMPLES = 50  # You can increase this

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
ambiguous_locs = [8, 18, 28, 38]  # Loc column indicating ambiguous hits
if "Loc" in df.columns:
    df = df[~df["Loc"].isin(ambiguous_locs)].reset_index(drop=True)
    print(f"Data filtered. Remaining rows: {len(df)}")

#df["is_ambiguous"] = df["Loc"].isin([8, 18, 28, 38]).astype(int)


#--------------TRAINING--------------------------
#------------------------------------------------
# Selecting features (all columns from "Distance_S1" to "Force_N")
feature_columns = df.loc[:, "ToA_S1":"Force_N"].columns
X = df[feature_columns]

# Selecting target variables (impact location coordinates)
y = df[["theta", "z"]]  



# Split data:
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.03, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# --------------------------------------
# PARAMETER GRID
# --------------------------------------
param_grid = {
    'n_estimators': [ 500, 600,700],
    'learning_rate': [0.015, 0.02, 0.025, 0.03],
    'max_depth': [6, 8, 10],
    'subsample': [0.4, 0.6, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0.6, 0.7, 0.8],
    'reg_lambda': [3.0, 3.5,4.0]
}

# Generate all param combinations and randomly sample
keys, values = zip(*param_grid.items())
all_combinations = [dict(zip(keys, v)) for v in product(*values)]
sampled_params = random.sample(all_combinations, min(NUM_RANDOM_SAMPLES, len(all_combinations)))

# --------------------------------------
# JOINT TUNING FUNCTION
# --------------------------------------
best_rmse = float("inf")
best_params = None
best_models = (None, None)

print(f"Testing {len(sampled_params)} random hyperparameter combinations...")

for i, params in enumerate(sampled_params):
    print(f"\n[{i+1}/{len(sampled_params)}] Testing: {params}")

    model_theta = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cuda", **params)
    model_z = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cuda", **params)


    model_theta.fit(X_train, y_train["theta"])
    model_z.fit(X_train, y_train["z"])

    pred_theta = model_theta.predict(X_val)
    pred_z = model_z.predict(X_val)

    spatial_error = np.sqrt(((y_val["theta"] - pred_theta) * r) ** 2 + (y_val["z"] - pred_z) ** 2)
    combined_rmse = np.mean(spatial_error)

    print(f"→ Spatial RMSE: {combined_rmse:.4f} cm")

    if combined_rmse < best_rmse:
        best_rmse = combined_rmse
        best_params = params
        best_models = (model_theta, model_z)

# --------------------------------------
# FINAL EVALUATION
# --------------------------------------
print("\n Best Params:")
print(best_params)
print(f"Best Spatial RMSE on val set: {best_rmse:.4f} cm")

model_theta, model_z = best_models
y_pred_theta = model_theta.predict(X_test)
y_pred_z = model_z.predict(X_test)

rmse_theta = np.sqrt(mean_squared_error(y_test["theta"], y_pred_theta)) * r
rmse_z = np.sqrt(mean_squared_error(y_test["z"], y_pred_z))
rmse_total = np.sqrt(np.mean(((y_test["theta"] - y_pred_theta)*r)**2 + (y_test["z"] - y_pred_z)**2))

print("\n====== FINAL TEST RESULTS ======")
print(f"RMSE (theta, cm): {rmse_theta:.4f}")
print(f"RMSE (z, cm): {rmse_z:.4f}")
print(f"Combined Spatial RMSE (cm): {rmse_total:.4f}")
