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


r = 11.55
z_max = 45
#---------------------------------------
error_distance = 3.5 # threshold
test_loc = "top"  # or "full"
#---------------------------------------

# data = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/Features_stlham_p1_tank.csv"
# df = pd.read_csv(data)

#folder_path = "/mnt/c/Users/tunta/OneDrive - Imperial College London/Y4 work/FYP/FYP_Data/Processed_Data/tank/16april"
folder_path = "16april"
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
# df["is_ambiguous"] = df["Loc"].isin(ambiguous_locs).astype(int)

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
# Always start with base features from ToA_S1 to S8_z
feature_columns = df.loc[:, "ToA_S1":"S8_z"].columns.tolist()

X = df[feature_columns]
# ------------------ Splitting ------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

test_locs = df.loc[y_test.index, "Loc"].reset_index(drop=True)

if test_loc == "top":
    top_locs = list(range(1, 16)) + list(range(31, 41))
    top_mask = test_locs.isin(top_locs)
    X_test = X_test.reset_index(drop=True).loc[top_mask]
    y_test = y_test.reset_index(drop=True).loc[top_mask]
    test_locs = test_locs.loc[top_mask].reset_index(drop=True)
    print(f" Test set: Top-only ({len(X_test)} samples)")
else:
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    test_locs = test_locs.reset_index(drop=True)
    print(f" Test set: Full ({len(X_test)} samples)")

# ------------------ XGBoost Parameters ------------------
xgb_params = {
    'objective': "reg:squarederror",
    'n_estimators': 600,  # Increase trees for better learning
    'learning_rate': 0.02,  # Lower learning rate for stability
    'max_depth': 10,  # Deeper trees for better feature learning
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'reg_lambda': 3,  # Regularization to prevent overfitting
    'reg_alpha': 0.5, # Regularization to prevent overfitting
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50,  # Early stopping to prevent overfitting
}
# ------------------ Training ------------------
xgb_model_sin = xgb.XGBRegressor(**xgb_params)
xgb_model_cos = xgb.XGBRegressor(**xgb_params)
xgb_model_z = xgb.XGBRegressor(**xgb_params)

xgb_model_sin.fit(X_train, y_train["sin_theta"], eval_set=[(X_val, y_val["sin_theta"])], verbose=False)
xgb_model_cos.fit(X_train, y_train["cos_theta"], eval_set=[(X_val, y_val["cos_theta"])], verbose=False)
xgb_model_z.fit(X_train, y_train["z"], eval_set=[(X_val, y_val["z"])], verbose=False)

# ------------------ Prediction ------------------
y_pred_sin = xgb_model_sin.predict(X_test)
y_pred_cos = xgb_model_cos.predict(X_test)
y_pred_z = xgb_model_z.predict(X_test)

# Reconstruct theta from sin and cos
pred_theta = np.arctan2(y_pred_sin, y_pred_cos)
true_theta = np.arctan2(y_test["sin_theta"], y_test["cos_theta"])
delta = pred_theta - true_theta
pred_theta = pred_theta - 2*np.pi * (delta > np.pi) + 2*np.pi * (delta < -np.pi)


# ------------------ Evaluation ------------------
rmse_theta = np.sqrt(mean_squared_error(true_theta, pred_theta)) * r
rmse_z = np.sqrt(mean_squared_error(y_test["z"], y_pred_z))
rmse_total = np.sqrt(np.mean(((true_theta - pred_theta) * r) ** 2 + (y_test["z"] - y_pred_z) ** 2))
rmse = np.array([rmse_theta, rmse_z, rmse_total])

# Compute spatial error (Euclidean distance in cm)
spatial_error = np.sqrt(((true_theta - pred_theta) * r) ** 2 + (y_test["z"] - y_pred_z) ** 2)

error_distance = 3.5  # cm threshold

TP_mask = spatial_error <= error_distance
FP_mask = ~TP_mask

true_positives = TP_mask.sum()
false_positives = FP_mask.sum()

accuracy = true_positives / len(y_test)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / len(y_test)

# ------------------ Summary ------------------

print("\n📊 Model Summary")
print(f"{'Metric':<30} | {'Theta':>10} | {'Z':>10}")
print("-" * 60)
# print(f"{'Best Iteration':<30} | {xgb_model_sin.best_iteration:>10} | {xgb_model_z.best_iteration:>10}")
# print(f"{'Best Val Score':<30} | {xgb_model_sin.best_score:>10.4f} | {xgb_model_z.best_score:>10.4f}")
print(f"{'RMSE (cm)':<30} | {rmse_theta:>10.4f} | {rmse_z:>10.4f}")
print("-" * 60)
print(f"{'Total Spatial RMSE (cm)':<30} | {rmse_total:>10.4f}")
print("-" * 60)
print(f"{'True Positives':<30}: {true_positives}")
print(f"{'False Positives':<30}: {false_positives}")
print(f"{'Accuracy':<30}: {accuracy:.4f}")
print(f"{'Precision':<30}: {precision:.4f}")
print(f"{'Recall':<30}: {recall:.4f}")
print("-" * 60)

# ------------------ Error Report ------------------
print(f"\n  Predictions with spatial error > {error_distance} cm:")
print("-" * 95)
print(f"{'Index':<6} {'True θ (rad)':<15} {'Pred θ (rad)':<15} {'True Z (cm)':<12} {'Pred Z (cm)':<12} {'Error (cm)':<10}")
print("-" * 95)

for i, (t, p, z_val, z_pred, err_cm) in enumerate(zip(true_theta, pred_theta, y_test["z"], y_pred_z, spatial_error)):
    if err_cm > error_distance:
        print(f"{i:<6} {t:+.4f}         {p:+.4f}         {z_val:<12.1f} {z_pred:<12.1f} {err_cm:.2f}")

#------------------------------------------------

## VISUALISATION
#plot_cylinder_predictions(y_test, y_pred_theta, y_pred_z)
plot_tankflatten_predictions(
    y_test=y_test,
    pred_theta=pred_theta,
    y_pred_z=y_pred_z,
    test_loc=test_loc,
    rmse_total=rmse_total,
    rmse_z=rmse_z,
    rmse_theta=rmse_theta,
    accuracy=accuracy,
    r=11.55,
    z_max=45,
    FP_mask=FP_mask
)

#plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model_sin, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances for Loc_sin")
#plt.show()
#plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model_cos, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances for Loc_cos")
#plt.show()

#plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model_z, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances for Loc_Z")
#plt.show()


# ------------------ IMPACT CLASSIFICATION ------------------
# ---------------------------------
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
start = time.time()
# ----------- Features & Target -----------
feature_columns = df.loc[:, "ToA_S1":"Force_N"].columns
X_cls = df[feature_columns]
y_cls = df["Impact_Type"]  # 0: soft, 1: hard

# ----------- Train/Test Split -----------
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# ----------- Train Classifier -----------
impact_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.02,
    max_depth=10,
    n_estimators=600,
    subsample=0.6,
    colsample_bytree=0.8,
    reg_lambda=3,
    reg_alpha=0.5,
    use_label_encoder=False
)

impact_clf.fit(X_train_cls, y_train_cls)

# ----------- Evaluate -----------
y_pred_cls = impact_clf.predict(X_test_cls)

end = time.time()
print(f"Runtime: {end - start:.4f} seconds")
print("Classification Report:")
print(classification_report(y_test_cls, y_pred_cls, digits=4))

# ----------- Confusion Matrix -----------
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test_cls, y_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Soft', 'Hard'], yticklabels=['Soft', 'Hard'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Soft vs Hard Impacts")
plt.tight_layout()
plt.show()

# ----------- Feature Importance -----------
xgb.plot_importance(impact_clf, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances (Impact Classification)")
plt.tight_layout()
plt.show()

#filename = 'predictions_XGB.mat'
# #save_predictions(y_test, y_pred_theta, y_pred_z, filename, rmse)



