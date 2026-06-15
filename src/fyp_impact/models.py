from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from .geometry import reconstructed_theta


@dataclass
class LocalisationModels:
    sin_model: xgb.XGBRegressor
    cos_model: xgb.XGBRegressor
    z_model: xgb.XGBRegressor

    def predict(self, x_values: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pred_sin = self.sin_model.predict(x_values)
        pred_cos = self.cos_model.predict(x_values)
        pred_z = self.z_model.predict(x_values)
        return reconstructed_theta(pred_sin, pred_cos), pred_z


def default_xgb_regressor_params(n_estimators: int = 600) -> dict[str, float | int | str]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": n_estimators,
        "learning_rate": 0.02,
        "max_depth": 10,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.5,
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": 42,
    }


def default_xgb_classifier_params(n_estimators: int = 600) -> dict[str, float | int | str]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": n_estimators,
        "learning_rate": 0.02,
        "max_depth": 10,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.5,
        "tree_method": "hist",
        "random_state": 42,
    }


def train_localisation_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: dict[str, object] | None = None,
) -> LocalisationModels:
    model_params = default_xgb_regressor_params()
    if params:
        model_params.update(params)

    sin_model = xgb.XGBRegressor(**model_params)
    cos_model = xgb.XGBRegressor(**model_params)
    z_model = xgb.XGBRegressor(**model_params)
    sin_model.fit(x_train, y_train["sin_theta"], eval_set=[(x_val, y_val["sin_theta"])], verbose=False)
    cos_model.fit(x_train, y_train["cos_theta"], eval_set=[(x_val, y_val["cos_theta"])], verbose=False)
    z_model.fit(x_train, y_train["z"], eval_set=[(x_val, y_val["z"])], verbose=False)
    return LocalisationModels(sin_model=sin_model, cos_model=cos_model, z_model=z_model)


def train_impact_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, object] | None = None,
) -> xgb.XGBClassifier:
    model_params = default_xgb_classifier_params()
    if params:
        model_params.update(params)

    classifier = xgb.XGBClassifier(**model_params)
    classifier.fit(x_train, y_train)
    return classifier
