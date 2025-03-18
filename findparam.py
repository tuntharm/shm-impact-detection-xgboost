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





# data = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/Features_stlham_p1_tank.csv"
# df = pd.read_csv(data)

folder_path = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/plate"

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
y = df[["Loc_X", "Loc_Y"]]  # Multi-output regression targets

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [450, 500, 550],  # Number of trees
    'learning_rate': [0.01, 0.02, 0.05, 0.1],  # Lower learning rates for stability
    'max_depth': [7, 8, 9],  # Control tree depth
    'subsample': [0.6, 0.7, 0.8],  # Control row sampling
    'colsample_bytree': [0.6, 0.7, 0.8],  # Feature sampling
    'reg_alpha': [0.3, 0.5, 0.7],  # L1 Regularization (Feature Selection)
    'reg_lambda': [2.0,2.5,3]  # L2 Regularization (Prevent Overfitting)
}

# Initialize XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

# Perform RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    scoring='neg_root_mean_squared_error',  # Minimize RMSE
    n_iter=20,  # Number of random combinations to try
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1,  # Use all available CPU cores
    random_state=42
)

# Fit RandomizedSearchCV (for Loc_X first)
random_search.fit(X_train, y_train["Loc_X"]) # Fit for Loc_X

# Get best parameters
best_params = random_search.best_params_
best_rmse = -random_search.best_score_

# Display the best parameters and RMSE
print("Best Parameters:", best_params)
print("Best RMSE:", best_rmse)
