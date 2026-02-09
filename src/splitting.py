from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, SPLIT

def random_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random row-wise split into train/val/test."""
    train_df, tmp_df = train_test_split(
        df, test_size=(1.0 - SPLIT.train), random_state=RANDOM_SEED, shuffle=True
    )
    # split remaining into val/test
    rel_test = SPLIT.test / (SPLIT.val + SPLIT.test)
    val_df, test_df = train_test_split(
        tmp_df, test_size=rel_test, random_state=RANDOM_SEED, shuffle=True
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def yearwise_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by whole calendar years (no year appears in more than one subset)."""
    years = np.array(sorted(df["Year"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(years)

    n = len(years)
    n_train = int(round(n * SPLIT.train))
    n_val = int(round(n * SPLIT.val))
    n_test = n - n_train - n_val
    # Ensure all non-empty if possible
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        # adjust to sum n
        while n_train + n_val + n_test > n:
            n_train = max(1, n_train - 1)
        while n_train + n_val + n_test < n:
            n_train += 1

    train_years = set(years[:n_train].tolist())
    val_years = set(years[n_train:n_train + n_val].tolist())
    test_years = set(years[n_train + n_val:].tolist())

    train_df = df[df["Year"].isin(train_years)].copy()
    val_df = df[df["Year"].isin(val_years)].copy()
    test_df = df[df["Year"].isin(test_years)].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
