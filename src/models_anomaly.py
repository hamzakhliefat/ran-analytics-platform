# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:22:50 2026

@author: hamza khlefat
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def label_by_quantile(df: pd.DataFrame, col: str, q: float) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    thr = float(s.quantile(q))
    return (s >= thr).astype(int)


def _prep_xy(df: pd.DataFrame, target: str, features: list[str]):
    use_features = [
        f for f in features
        if isinstance(f, str) and (f in df.columns) and (f != target)
    ]
    use_cols = [target] + use_features

    d = df[use_cols].copy()

    d[target] = pd.to_numeric(d[target], errors="coerce").fillna(0).astype(int)
    for f in use_features:
        d[f] = pd.to_numeric(d[f], errors="coerce")

    d = d.dropna(subset=[target])
    X = d[use_features].copy()
    if X.empty:
        return None, None, None, None

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = d[target].astype(int)

    return d, X, y, use_features


def _clf_metrics(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "test_F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            out["test_ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["test_ROC_AUC"] = np.nan
    else:
        out["test_ROC_AUC"] = np.nan
    return out


def train_classifiers(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = 0.2,   
    cv_folds: int = 5,        
    random_state: int = 42,
):
    """
    Train 3+ classifiers, evaluate on test, run CV F1, pick best by highest test_F1.
    """
    _, X, y, used_features = _prep_xy(df, target, features)
    if X is None:
        raise ValueError("No usable features after preprocessing.")

    # Stratified partitioning for class distribution preservation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced"
        ),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rows = []
    fitted = {}

    for name, model in models.items():
        # CV F1
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, scoring="f1", cv=cv, n_jobs=-1
            )
            cv_f1 = float(np.mean(cv_scores))
        except Exception:
            cv_f1 = np.nan

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None

        m = _clf_metrics(y_test, y_pred, y_prob=y_prob)
        rows.append({"model": name, **m, "cv_F1": cv_f1})
        fitted[name] = model

    results_df = (
        pd.DataFrame(rows)
        .sort_values("test_F1", ascending=False)
        .reset_index(drop=True)
    )

    best_name = results_df.loc[0, "model"]
    best_model = fitted[best_name]

    cm = confusion_matrix(y_test, best_model.predict(X_test))

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "results_df": results_df,
        "confusion_matrix": cm,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": used_features,
    }


def shap_explain_classifier(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_samples: int = 400,
    random_state: int = 42,
):
    """
    SHAP for binary classification using probability of class=1.
    Returns Explanation object suitable for plots.
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

    # explain probability of anomaly (class 1)
    def _predict_proba_1(X):
        return model.predict_proba(X)[:, 1]

    try:
        explainer = shap.Explainer(_predict_proba_1, xb)
        shap_values = explainer(xe)
        return {"ok": True, "explainer": explainer, "shap_values": shap_values, "X_used": xe}
    except Exception as e:
        return {"ok": False, "error": str(e)}
