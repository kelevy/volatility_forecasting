"""
Downloads S&P 500 daily price data and computes log returns and realized
volatility (the forecasting target). Caches to disk so repeated runs don't
re-download.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "sp500_raw.csv")

TICKER = "^GSPC"
START_DATE = "2005-01-01"


def download_data(ticker: str = TICKER, start: str = START_DATE) -> pd.DataFrame:
    """Download OHLC data via yfinance and cache locally."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
        return df

    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(
            f"No data downloaded for {ticker}. Check ticker/internet connection."
        )
    df.to_csv(RAW_PATH)
    return df


def add_returns_and_volatility(df: pd.DataFrame, vol_window: int = 5) -> pd.DataFrame:
    """
    Adds:
      - log_return: daily log return
      - realized_vol: forward-looking annualized realized volatility over the
        next `vol_window` trading days (THIS IS THE FORECAST TARGET, and it is
        deliberately forward-looking — it must never be used as a feature,
        only as the label `y`).
      - realized_vol_lag1: yesterday's realized_vol computed over the
        TRAILING `vol_window` days — this is a legitimate feature.
    """
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Forward-looking target: volatility over the NEXT vol_window days.
    # Uses .shift(-vol_window) intentionally -- this is the label, not a feature.
    trailing_std = df["log_return"].rolling(vol_window).std()
    df["realized_vol"] = trailing_std.shift(-vol_window) * np.sqrt(252)

    # Legitimate lagged feature: trailing (backward-looking) realized vol.
    df["realized_vol_lag1"] = trailing_std * np.sqrt(252)

    df = df.dropna(subset=["log_return"])
    return df


def load_dataset(vol_window: int = 5) -> pd.DataFrame:
    raw = download_data()
    return add_returns_and_volatility(raw, vol_window=vol_window)


if __name__ == "__main__":
    data = load_dataset()
    print(data.tail())
    print(f"\nRows: {len(data)}")