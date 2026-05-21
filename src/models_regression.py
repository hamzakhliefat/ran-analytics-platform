# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 11:40:19 2026

@author: hamza khlefat
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def _prep_xy(df: pd.DataFrame, target: str, features: list[str]):
    if not isinstance(target, str):
        raise ValueError(f"target must be a string column name, got: {type(target)}")

    # Remove target from features to prevent circular leakage
    use_features = [
        f for f in features
        if isinstance(f, str) and (f in df.columns) and (f != target)
    ]

    if target not in df.columns:
        return None, None, None, None

    use_cols = [target] + use_features
    d = df[use_cols].copy()

    # numeric coercion
    d[target] = pd.to_numeric(d[target], errors="coerce")
    for f in use_features:
        d[f] = pd.to_numeric(d[f], errors="coerce")

    d = d.dropna(subset=[target])

    X = d[use_features].copy()
    if X.empty:
        return None, None, None, None

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    y = d[target].astype(float)
    return d, X, y, use_features


def _reg_metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))

    # sklearn 1.7+ removed squared=False in some contexts → keep safe:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    r2 = float(r2_score(y_true, y_pred))
    return {"test_MAE": mae, "test_RMSE": rmse, "test_R2": r2}


def train_regressors(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = 0.2,   
    cv_folds: int = 5,        
    random_state: int = 42,
):
    """
    Train 3+ regression models, evaluate on test, run CV, pick best by lowest test_MAE.
    Returns a pack used by Streamlit page.
    """
    _, X, y, used_features = _prep_xy(df, target, features)
    if X is None:
        raise ValueError("No usable features after preprocessing (check columns & missing values).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "LinearRegression": LinearRegression(),
        "ElasticNet(alpha=0.1, l1_ratio=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rows = []
    fitted = {}

    for name, model in models.items():
        # CV MAE (negative -> positive)
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=-1
            )
            cv_mae = float(-np.mean(cv_scores))
        except Exception:
            cv_mae = np.nan

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = _reg_metrics(y_test, pred)

        rows.append({"model": name, **m, "cv_MAE": cv_mae})
        fitted[name] = model

    results_df = (
        pd.DataFrame(rows)
        .sort_values("test_MAE", ascending=True)
        .reset_index(drop=True)
    )

    best_name = results_df.loc[0, "model"]
    best_model = fitted[best_name]

    preds_df = pd.DataFrame(
        {"y_true": y_test.values, "y_pred": best_model.predict(X_test)}
    )

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "results_df": results_df,
        "preds_df": preds_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": used_features,
    }


def shap_explain_regression(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_samples: int = 400,
    random_state: int = 42,
):
    """
    Returns {"ok": bool, "explainer":..., "shap_values":..., "X_used":..., "error":...}
    Produces SHAP Explanation object suitable for plots.
    """
    try:
        import shap
    except Exception as e:
        return {"ok": False, "error": f"SHAP is not installed: {e}"}

    xb = X_background.copy()
    xe = X_explain.copy()

    xb = xb.replace([np.inf, -np.inf], np.nan).fillna(xb.median(numeric_only=True))
    xe = xe.replace([np.inf, -np.inf], np.nan).fillna(xe.median(numeric_only=True))

    if len(xb) > max_samples:
        xb = xb.sample(n=max_samples, random_state=random_state)
    if len(xe) > max_samples:
        xe = xe.sample(n=max_samples, random_state=random_state)

    # Structured explainer initialization for multi-architecture support
    try:
        if hasattr(model, "estimators_") or model.__class__.__name__ in ["RandomForestRegressor", "GradientBoostingRegressor"]:
            explainer = shap.TreeExplainer(model, data=xb, feature_perturbation="interventional")
            shap_values = explainer(xe, check_additivity=False)  # avoid additivity crash
        else:
            explainer = shap.Explainer(model, xb)
            shap_values = explainer(xe)

        return {"ok": True, "explainer": explainer, "shap_values": shap_values, "X_used": xe}
    except Exception as e:
        return {"ok": False, "error": str(e)}
