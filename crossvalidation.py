from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os
import glob
from xgboost.callback import EarlyStopping
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"





from plot_utils import plot_cylinder_predictions
from plot_utils import plot_tankflatten_predictions
from savefile import save_predictions
import time

start = time.time()
r = 11.55
z_max = 45
#---------------------------------------
error_distance = 3.5 # threshold
test_loc = "top"  # or "full"
n_splits = 10
#---------------------------------------

# data = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/Features_stlham_p1_tank.csv"
# df = pd.read_csv(data)

folder_path = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/tank/16april"

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

# ---------------------- AMBIGUOUS LOCATIONS ----------------------
# Define ambiguous locations once
ambiguous_locs = [8, 18, 28, 38]

# ---------------------- MARK AMBIGUOUS ----------------------
# Uncomment to mark ambiguous samples
df["is_ambiguous"] = df["Loc"].isin(ambiguous_locs).astype(int)

# ---------------------- FILTER AMBIGUOUS ROWS ----------------------
# Uncomment to remove ambiguous samples
# df = df[df["Loc"].isin(ambiguous_locs) == False].reset_index(drop=True)
# print(f"Ambiguous rows removed. Remaining rows: {len(df)}")



# Define correct angular and axial positions
sensor_theta = [0, np.pi/2, np.pi, 3*np.pi/2] * 2
sensor_z = [0]*4 + [45]*4

# Add theta and z features for each sensor
for i in range(8):
    df[f"S{i+1}_theta"] = sensor_theta[i]
    df[f"S{i+1}_z"] = sensor_z[i]

#--------------TRAINING--------------------------
#------------------------------------------------

# -------------------------- TRAINING SETUP --------------------------
# ---------------------- TARGET FEATURES ---------------------
df["sin_theta"] = np.sin(df["theta"])
df["cos_theta"] = np.cos(df["theta"])
y = df[["sin_theta", "cos_theta", "z"]]

# ---------------------- INPUT FEATURES ----------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------- Parameters -----------
test_loc = "full"  # or "top"

# ----------- Features & Target -----------
feature_columns = df.loc[:, "ToA_S1":"S8_z"].columns.drop("Impact_Type", errors="ignore")
X_cls = df[feature_columns]
y_cls = df["Impact_Type"]  # 0: soft, 1: hard

# ----------- Train/Test Split -----------
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# ----------- Loc-Based Filtering -----------
test_locs_cls = df.loc[y_test_cls.index, "Loc"].reset_index(drop=True)

if test_loc == "top":
    top_locs = list(range(1, 16)) + list(range(31, 41))
    top_mask_cls = test_locs_cls.isin(top_locs)

    X_test_cls = X_test_cls.reset_index(drop=True).loc[top_mask_cls]
    y_test_cls = y_test_cls.reset_index(drop=True).loc[top_mask_cls]
    test_locs_cls = test_locs_cls.loc[top_mask_cls].reset_index(drop=True)

    print(f"Classification test set: Top-only ({len(X_test_cls)} samples)")
else:
    X_test_cls = X_test_cls.reset_index(drop=True)
    y_test_cls = y_test_cls.reset_index(drop=True)
    test_locs_cls = test_locs_cls.reset_index(drop=True)

    print(f"Classification test set: Full ({len(X_test_cls)} samples)")

# ----------- Build ANN Model -----------
def build_classifier(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # binary output
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model_cls = build_classifier(X_train_cls.shape[1])

# ----------- Train Model -----------
model_cls.fit(X_train_cls, y_train_cls, epochs=100, batch_size=32, verbose=0)

# ----------- Evaluate Model -----------
y_pred_probs = model_cls.predict(X_test_cls, verbose=0).flatten()
y_pred_cls = (y_pred_probs >= 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test_cls, y_pred_cls, digits=4))
end = time.time()
print(f"Time taken for classification: {end - start:.4f} seconds")
# ----------- Confusion Matrix -----------
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test_cls, y_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Soft', 'Hard'], yticklabels=['Soft', 'Hard'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Soft vs Hard Impacts")
plt.tight_layout()
plt.show()
