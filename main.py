"""
Runs the full volatility forecasting pipeline:
  1. Load and prepare S&P 500 data
  2. Build the leak-free feature matrix
  3. Run walk-forward validation for all four models:
       - Naive baseline (yesterday's realized vol = forecast)
       - GARCH(1,1) classical econometric baseline
       - XGBoost (ML, hand-engineered features)
       - LSTM (DL, raw return sequences)
  4. Evaluate with RMSE, QLIKE, and directional accuracy
  5. Print results table and save diagnostic plots to results/
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_dataset
from features import build_feature_matrix, get_feature_names, TARGET_COL
from validation import walk_forward_splits, split_xy
from evaluate import evaluate_all, results_table

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_SPLITS = 5
VOL_WINDOW = 5
MIN_TRAIN_SIZE = 0.6


# ─────────────────────────────────────────────
# Naive baseline
# ─────────────────────────────────────────────

def run_naive(feature_df, folds):
    """
    Naive baseline: predict yesterday's realized vol as today's forecast.
    Always included as the minimum sanity floor — if a model can't beat
    this, that's a finding worth reporting.
    """
    all_preds, all_actuals, all_prev = [], [], []
    for fold in folds:
        y_test = feature_df.iloc[fold.test_idx][TARGET_COL].values
        y_lag = feature_df.iloc[fold.test_idx]["realized_vol_lag1"].values
        all_preds.append(y_lag)
        all_actuals.append(y_test)
        all_prev.append(y_lag)
    return np.concatenate(all_preds), np.concatenate(all_actuals), np.concatenate(all_prev)


# ─────────────────────────────────────────────
# GARCH walk-forward
# ─────────────────────────────────────────────

def run_garch(feature_df, raw_df, folds):
    from models.baseline_garch import GarchBaseline

    all_preds, all_actuals, all_prev = [], [], []
    log_returns = raw_df["log_return"].reindex(feature_df.index).dropna()

    for fold in folds:
        test_dates = feature_df.index[fold.test_idx]
        train_dates = feature_df.index[fold.train_idx]

        train_returns = log_returns.loc[log_returns.index.isin(train_dates)] * 100
        if len(train_returns) < 50:
            continue

        model = GarchBaseline(horizon=VOL_WINDOW).fit(train_returns)
        forecast = model.forecast_annualized_vol()

        y_test = feature_df.iloc[fold.test_idx][TARGET_COL].values
        y_prev = feature_df.iloc[fold.test_idx]["realized_vol_lag1"].values

        # GARCH gives one forecast per fold; broadcast across the test window
        all_preds.append(np.full(len(y_test), forecast))
        all_actuals.append(y_test)
        all_prev.append(y_prev)

    return np.concatenate(all_preds), np.concatenate(all_actuals), np.concatenate(all_prev)


# ─────────────────────────────────────────────
# XGBoost walk-forward
# ─────────────────────────────────────────────

def run_xgb(feature_df, folds):
    from models.ml_xgboost import XGBVolModel

    feature_cols = get_feature_names(feature_df)
    all_preds, all_actuals, all_prev = [], [], []

    for fold in folds:
        X_train, y_train, X_test, y_test = split_xy(
            feature_df, feature_cols, TARGET_COL, fold
        )
        model = XGBVolModel().fit(X_train, y_train)
        preds = model.predict(X_test)
        y_prev = feature_df.iloc[fold.test_idx]["realized_vol_lag1"].values

        all_preds.append(preds)
        all_actuals.append(y_test.values)
        all_prev.append(y_prev)

    return np.concatenate(all_preds), np.concatenate(all_actuals), np.concatenate(all_prev)


# ─────────────────────────────────────────────
# LSTM walk-forward
# ─────────────────────────────────────────────

def run_lstm(feature_df, raw_df, folds):
    from models.dl_lstm import run_lstm_walk_forward

    log_returns = raw_df["log_return"].reindex(feature_df.index)
    target = feature_df[TARGET_COL]
    return run_lstm_walk_forward(log_returns, target, folds, seq_len=21)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Volatility Forecasting: GARCH vs XGBoost vs LSTM")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading S&P 500 data...")
    raw_df = load_dataset(vol_window=VOL_WINDOW)
    print(f"      {len(raw_df)} trading days loaded "
          f"({raw_df.index[0].date()} → {raw_df.index[-1].date()})")

    # 2. Build features
    print("\n[2/5] Building feature matrix...")
    feature_df = build_feature_matrix(raw_df)
    feature_df = feature_df.dropna(subset=[TARGET_COL])
    print(f"      {len(feature_df)} rows after feature engineering, "
          f"{len(get_feature_names(feature_df))} features")

    # 3. Build walk-forward folds
    print(f"\n[3/5] Setting up {N_SPLITS}-fold walk-forward validation...")
    folds = list(walk_forward_splits(len(feature_df), n_splits=N_SPLITS,
                                     min_train_size=MIN_TRAIN_SIZE))
    for fold in folds:
        print(f"      Fold {fold.fold_number}: "
              f"train={len(fold.train_idx)} rows, test={len(fold.test_idx)} rows")

    # 4. Run models
    print("\n[4/5] Running models...")
    all_results = {}
    all_forecasts = {}

    # Naive
    print("      → Naive baseline...")
    naive_preds, actuals, naive_prev = run_naive(feature_df, folds)
    all_results["Naive"] = evaluate_all(actuals, naive_preds, naive_prev)
    all_forecasts["Naive"] = naive_preds

    # GARCH
    print("      → GARCH(1,1)...")
    try:
        garch_preds, garch_actuals, garch_prev = run_garch(feature_df, raw_df, folds)
        all_results["GARCH"] = evaluate_all(garch_actuals, garch_preds, garch_prev)
        all_forecasts["GARCH"] = garch_preds
    except ImportError:
        print("        [skipped — install `arch` package]")

    # XGBoost
    print("      → XGBoost...")
    try:
        xgb_preds, xgb_actuals, xgb_prev = run_xgb(feature_df, folds)
        all_results["XGBoost"] = evaluate_all(xgb_actuals, xgb_preds, xgb_prev)
        all_forecasts["XGBoost"] = xgb_preds
    except ImportError:
        print("        [skipped — install `xgboost` package]")

    # LSTM
    print("      → LSTM...")
    try:
        lstm_preds, lstm_actuals, lstm_prev = run_lstm(feature_df, raw_df, folds)
        all_results["LSTM"] = evaluate_all(lstm_actuals, lstm_preds, lstm_prev)
        all_forecasts["LSTM"] = lstm_preds
    except ImportError:
        print("        [skipped — install `torch` package]")

    # 5. Results
    print("\n[5/5] Results")
    print("=" * 60)
    table = results_table(all_results)
    print(table.to_string(float_format="{:.6f}".format))
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "results.csv"))
    print("Saved: results/results.csv")

    # Save forecasts for use in notebooks
    forecasts_df = pd.DataFrame(all_forecasts)
    forecasts_df["actual"] = actuals
    forecasts_df.to_csv(os.path.join(RESULTS_DIR, "forecasts.csv"), index=False)
    print("Saved: results/forecasts.csv")

    print("\nDone.")
