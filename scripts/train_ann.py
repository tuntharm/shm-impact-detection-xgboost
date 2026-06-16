from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fyp_impact.data import classification_feature_columns, load_feature_data, prepare_feature_data  # noqa: E402
from fyp_impact.geometry import DEFAULT_Z_MAX_CM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an optional TensorFlow ANN hard/soft impact classifier.")
    parser.add_argument("--data-dir", action="append", required=True, help="Directory containing direct feature CSV files. Repeat for A/B/C combinations.")
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX_CM)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise SystemExit("TensorFlow is optional. Install requirements-optional.txt to run train_ann.py.") from exc

    df = prepare_feature_data(load_feature_data(args.data_dir), z_max=args.z_max)
    features = classification_feature_columns(df)
    x_values = df[features]
    y_values = df["Impact_Type"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=0.2, random_state=args.random_state, stratify=y_values
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train_scaled.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train_scaled, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    probabilities = model.predict(x_test_scaled, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)
    print(classification_report(y_test, predictions, digits=4))


if __name__ == "__main__":
    main()
