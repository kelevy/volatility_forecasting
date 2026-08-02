"""
Leak-free feature engineering for the ML/DL models.

RULE ENFORCED THROUGHOUT: every feature at row t must be computable using only
information available AT OR BEFORE t. Any feature using .shift(-k) (a negative
/ forward shift) is a bug, except for the target column itself which is
deliberately excluded from the feature set (see build_feature_matrix).
"""

import numpy as np
import pandas as pd

TARGET_COL = "realized_vol"
FORBIDDEN_IN_FEATURES = {TARGET_COL}  # never allow the label to leak in as a feature


def build_feature_matrix(df: pd.DataFrame, n_lags: int = 10) -> pd.DataFrame:
    """
    Builds a leak-free feature set:
      - lagged log returns (t-1 ... t-n_lags)
      - lagged realized volatility (trailing, computed in data_loader)
      - rolling mean/std of returns over multiple trailing windows
      - simple momentum feature (trailing cumulative return)

    Returns a DataFrame with features + the target column, NaNs dropped.
    """
    df = df.copy()

    for lag in range(1, n_lags + 1):
        df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    for window in (5, 10, 21):
        df[f"ret_roll_mean_{window}"] = df["log_return"].shift(1).rolling(window).mean()
        df[f"ret_roll_std_{window}"] = df["log_return"].shift(1).rolling(window).std()

    df["momentum_21"] = df["log_return"].shift(1).rolling(21).sum()

    feature_cols = [c for c in df.columns if c.startswith(("ret_lag_", "ret_roll_", "momentum_"))]
    feature_cols += ["realized_vol_lag1"]

    _assert_no_leakage(feature_cols)

    keep = feature_cols + [TARGET_COL]
    out = df[keep].dropna()
    return out


def _assert_no_leakage(feature_cols):
    """Defensive check: make sure the target never sneaks into the feature list."""
    leaked = FORBIDDEN_IN_FEATURES.intersection(feature_cols)
    if leaked:
        raise ValueError(f"Look-ahead bias detected: {leaked} present in feature columns.")


def get_feature_names(feature_df: pd.DataFrame):
    return [c for c in feature_df.columns if c != TARGET_COL]


if __name__ == "__main__":
    from data_loader import load_dataset

    data = load_dataset()
    feats = build_feature_matrix(data)
    print(feats.head())
    print(f"\nFeature columns: {get_feature_names(feats)}")
    print(f"Rows after feature engineering: {len(feats)}")