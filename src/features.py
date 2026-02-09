from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encode month (1..12) into sin/cos features."""
    df = df.copy()
    theta = 2 * np.pi * (df["Month"].astype(float) - 1.0) / 12.0
    df["Month_sin"] = np.sin(theta)
    df["Month_cos"] = np.cos(theta)
    return df

def make_feature_matrix(df: pd.DataFrame, include_month_cyclic: bool = True) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names)."""
    work = df.copy()
    base_feats = ["ENSO", "NAO", "TSA", "TNA"]
    feat_cols = base_feats[:]
    if include_month_cyclic:
        work = add_cyclic_month(work)
        feat_cols += ["Month_sin", "Month_cos"]
    X = work[feat_cols].to_numpy(dtype=float)
    return X, feat_cols

def fit_feature_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def transform_features(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)
