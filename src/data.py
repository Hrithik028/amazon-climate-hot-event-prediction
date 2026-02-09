from __future__ import annotations

import pandas as pd

def load_temperature_data(path: str) -> pd.DataFrame:
    """Load monthly temperature + climate indices data.

    Expected columns (typical): Year, Month, Temp, ENSO, NAO, TSA, TNA
    (Actual column names can vary slightly — adjust in `standardise_columns`.)
    """
    df = pd.read_csv(path)
    return standardise_columns(df)

def load_thresholds(path: str) -> pd.DataFrame:
    """Load month-specific temperature thresholds."""
    thr = pd.read_csv(path)
    return standardise_thresholds(thr)

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Make a conservative best-effort mapping for common column names.
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"year"}:
            rename_map[c] = "Year"
        elif cl in {"month"}:
            rename_map[c] = "Month"
        elif cl in {"temp", "temperature", "t"}:
            rename_map[c] = "Temp"
        elif cl in {"enso"}:
            rename_map[c] = "ENSO"
        elif cl in {"nao"}:
            rename_map[c] = "NAO"
        elif cl in {"tsa"}:
            rename_map[c] = "TSA"
        elif cl in {"tna"}:
            rename_map[c] = "TNA"
    df = df.rename(columns=rename_map)

    required = {"Year", "Month", "Temp", "ENSO", "NAO", "TSA", "TNA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. Present: {list(df.columns)}"
        )
    df = df.sort_values(["Year", "Month"]).reset_index(drop=True)
    return df

def standardise_thresholds(thr: pd.DataFrame) -> pd.DataFrame:
    # Expected: Month, Threshold (or similar)
    rename_map = {}
    for c in thr.columns:
        cl = c.strip().lower()
        if cl in {"month"}:
            rename_map[c] = "Month"
        elif cl in {"threshold", "temp_threshold", "temperature_threshold"}:
            rename_map[c] = "Threshold"
    thr = thr.rename(columns=rename_map)
    if not {"Month", "Threshold"} <= set(thr.columns):
        raise ValueError(f"Threshold file must contain Month and Threshold columns. Got {thr.columns}")
    return thr[["Month", "Threshold"]].copy()

def add_hot_label(df: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Add binary 'Hot' label: 1 if Temp exceeds month-specific threshold, else 0."""
    thr = thresholds.set_index("Month")["Threshold"]
    df = df.copy()
    df["Threshold"] = df["Month"].map(thr)
    if df["Threshold"].isna().any():
        missing_months = sorted(df.loc[df["Threshold"].isna(), "Month"].unique().tolist())
        raise ValueError(f"Missing thresholds for months: {missing_months}")
    df["Hot"] = (df["Temp"] > df["Threshold"]).astype(int)
    return df
