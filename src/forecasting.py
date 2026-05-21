# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:12:45 2026

@author: hamza khlefat
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _find_time_col(df: pd.DataFrame) -> str | None:
    candidates = ["date", "day", "timestamp", "time", "dt", "datetime", "event_time"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first datetime-like col
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None


def prepare_daily_series(
    df: pd.DataFrame,
    target_col: str,
    cell_col: str | None = "cell_id",
    selected_cell: str | int | None = None,
) -> tuple[pd.Series, dict]:
    """
    Returns (daily_series, meta). daily_series index=DatetimeIndex (daily).
    """
    meta = {"time_col": None, "cell_used": None, "notes": []}
    tcol = _find_time_col(df)
    if tcol is None:
        raise ValueError("No time/date column found. Add a datetime column (e.g., date/timestamp).")

    d = df.copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna(subset=[tcol])

    if target_col not in d.columns:
        raise ValueError(f"Target column missing: {target_col}")

    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d = d.dropna(subset=[target_col])

    # optional cell filter
    if cell_col and (cell_col in d.columns) and (selected_cell is not None):
        d = d[d[cell_col].astype(str) == str(selected_cell)]
        meta["cell_used"] = str(selected_cell)

    meta["time_col"] = tcol

    # daily aggregation
    d["__day__"] = d[tcol].dt.floor("D")
    s = d.groupby("__day__")[target_col].mean().sort_index()

    # fill missing dates
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    s = s.interpolate(limit_direction="both")
    s.name = target_col
    return s, meta


def prepare_multivariate_series(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    cell_col: str | None = "cell_id",
    selected_cell: str | int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (daily_df, meta). daily_df index=DatetimeIndex (daily).
    Includes target and multiple features.
    """
    meta = {"time_col": None, "cell_used": None, "features_used": feature_cols}
    tcol = _find_time_col(df)
    if tcol is None:
        raise ValueError("No time/date column found.")

    d = df.copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna(subset=[tcol])

    all_cols = [target_col] + feature_cols
    for c in all_cols:
        if c not in d.columns:
            raise ValueError(f"Column missing: {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    
    d = d.dropna(subset=all_cols)

    # optional cell filter
    if cell_col and (cell_col in d.columns) and (selected_cell is not None):
        d = d[d[cell_col].astype(str) == str(selected_cell)]
        meta["cell_used"] = str(selected_cell)

    meta["time_col"] = tcol

    # daily aggregation
    d["__day__"] = d[tcol].dt.floor("D")
    df_agg = d.groupby("__day__")[all_cols].mean().sort_index()

    # fill missing dates
    full_idx = pd.date_range(df_agg.index.min(), df_agg.index.max(), freq="D")
    df_agg = df_agg.reindex(full_idx)
    df_agg = df_agg.interpolate(limit_direction="both")
    
    return df_agg, meta


def forecast_baseline(series: pd.Series, horizon: int = 7) -> pd.DataFrame:
    """
    Simple strong baseline: last-7 mean + trend using linear regression on last 30 days.
    Returns df with columns: ds, y, yhat (future).
    """
    s = series.dropna().copy()
    s = s.astype(float)

    # trend fit on last 30
    tail = s.tail(30)
    x = np.arange(len(tail))
    y = tail.values
    if len(tail) < 5:
        slope = 0.0
        intercept = float(s.iloc[-1])
    else:
        slope, intercept = np.polyfit(x, y, deg=1)

    last_val = float(s.iloc[-1])
    last7_mean = float(s.tail(7).mean())

    future_idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    x_future = np.arange(len(tail), len(tail) + horizon)
    trend = intercept + slope * x_future

    # blend (smooth + trend)
    yhat = 0.6 * trend + 0.4 * last7_mean
    yhat = np.maximum(yhat, 0.0)  # throughput can't be negative

    hist = pd.DataFrame({"ds": s.index, "y": s.values})
    fut = pd.DataFrame({"ds": future_idx, "yhat": yhat})
    return hist, fut


def forecast_lstm(
    data: pd.Series | pd.DataFrame, 
    horizon: int = 7, 
    lookback: int = 30, 
    epochs: int = 10
) -> pd.DataFrame:
    """
    Enhanced DL: Supports both univariate (Series) and multivariate (DataFrame) forecasting.
    Input 'data' should have target column as the first column if it is a DataFrame.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras import backend as K
        import gc
    except Exception as e:
        raise RuntimeError(f"TensorFlow initialization failure: {e}")

    # Convert Series to DataFrame for uniform handling
    if isinstance(data, pd.Series):
        df_vals = data.to_frame()
    else:
        df_vals = data.copy()

    s = df_vals.dropna().astype(float)
    if len(s) < lookback + 5:
        raise ValueError("Insufficient historical data points for neural network convergence.")

    # Dataset windowing for computational efficiency
    if len(s) > 365:
        s = s.tail(365)

    # Feature Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(s.values)
    
    n_features = scaled_data.shape[1]

    def make_xy(arr, lb):
        X, y = [], []
        for i in range(lb, len(arr)):
            X.append(arr[i - lb : i])
            y.append(arr[i, 0]) # Target is always col 0
        return np.array(X), np.array(y)

    X, y = make_xy(scaled_data, lookback)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    try:
        # Recurrent state reset and garbage collection to optimize memory utilization
        K.clear_session()
        gc.collect()
        
        # Architecture definition: Streamlined recurrent network supporting N features
        model = Sequential([
            LSTM(16, input_shape=(lookback, n_features)),  
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        
        # Inference execution with optimized batching
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)  

        # iterative multi-step forecast
        last_window = scaled_data[-lookback:].copy()
        preds = []
        w = last_window
        
        for _ in range(horizon):
            # Shape for prediction: (1, lookback, n_features)
            p = model.predict(w.reshape(1, lookback, n_features), verbose=0)[0, 0]
            preds.append(p)
            
            # For multivariate, we keep features constant for future steps (simplified assumption)
            # or we could use the last known values. Here we append the target prediction.
            next_row = w[-1].copy()
            next_row[0] = p # Update predicted target
            w = np.vstack([w[1:], [next_row]])

        # Inverse scaling for the target (col 0)
        # We need a dummy array to inverse scale correctly
        dummy = np.zeros((len(preds), n_features))
        dummy[:, 0] = preds
        inv_preds = scaler.inverse_transform(dummy)[:, 0]
        
        yhat = np.maximum(inv_preds, 0.0)

        future_idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        hist = pd.DataFrame({"ds": s.index, "y": s.iloc[:, 0].values})
        fut = pd.DataFrame({"ds": future_idx, "yhat": yhat})
        
        return hist, fut
        
    finally:
        # Post-inference resource deallocation
        try:
            K.clear_session()
            del model
            gc.collect()
        except:
            pass

