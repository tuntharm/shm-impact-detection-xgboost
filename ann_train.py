import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from plot_utils import plot_plate_predictions

# -------------- LOADING DATA --------------------------
folder_path = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/plate"

# Get all CSV files in the folder (ignoring subfolders)
csv_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if os.path.isfile(f)]

# Check if files exist
if not csv_files:
    raise FileNotFoundError("No CSV files found in the specified folder.")

# Load all CSV files into a single DataFrame
dfs = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dfs, ignore_index=True)

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Selecting features (from "ToA_S1" to "Force_N")
feature_columns = df.loc[:, "ToA_S1":"Force_N"].columns
X = df[feature_columns]
y = df[["Loc_X", "Loc_Y"]]  # Multi-output regression targets

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -------------- DEFINE ANN MODEL --------------------------
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)  # Output layer (predicting Loc_X, Loc_Y)
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# Train Model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=200, 
    batch_size=16, 
    verbose=1
)

# -------------- MAKE PREDICTIONS --------------------------
y_pred_ann = model.predict(X_test)

# Compute RMSE
rmse_x_ann = np.sqrt(mean_squared_error(y_test["Loc_X"], y_pred_ann[:, 0]))
rmse_y_ann = np.sqrt(mean_squared_error(y_test["Loc_Y"], y_pred_ann[:, 1]))
rmse_total_ann = np.sqrt(
    np.mean(
        (y_test["Loc_X"] - y_pred_ann[:, 0]) ** 2 +
        (y_test["Loc_Y"] - y_pred_ann[:, 1]) ** 2
    )
)

print(f"RMSE for Loc_X (ANN): {rmse_x_ann:.4f}")
print(f"RMSE for Loc_Y (ANN): {rmse_y_ann:.4f}")
print(f"Total RMSE (ANN): {rmse_total_ann:.4f}")

# -------------- VISUALIZE TRAINING LOSS --------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('ANN Training Progress')
plt.legend()
plt.show()


plot_plate_predictions(y_test, y_pred_ann[:, 0], y_pred_ann[:, 1]) 
