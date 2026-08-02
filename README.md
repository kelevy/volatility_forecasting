# Volatility Forecasting: GARCH vs. Gradient Boosting vs. LSTM

A research-style comparison of three approaches to forecasting realized volatility
on the S&P 500: a classical statistical baseline, a modern ML model, and a
deep learning model, evaluated under a walk-forward (no look-ahead) validation
scheme with financially meaningful loss functions.

## Motivation

Volatility forecasting is a core problem in quantitative finance — used in
options pricing, risk management, and position sizing. This project asks:
**does adding model complexity (ML, then DL) actually improve volatility
forecasts over a classical econometric baseline, once you evaluate honestly?**


## Methodology

### 1. Data
- S&P 500 (`^GSPC`) daily OHLC data via `yfinance`, ~15+ years of history.
- Target: realized volatility, computed as the rolling standard deviation of
  log returns over the next `h` trading days (forecast horizon), annualized.

### 2. Models compared
| Model | Type | Role |
|---|---|---|
| GARCH(1,1) | Classical econometric | Baseline — the industry-standard model for volatility clustering |
| XGBoost | Gradient-boosted trees | ML model using engineered features (lagged realized vol, returns, technical indicators) |
| LSTM | Deep learning | Sequence model learning directly from the return/volatility time series |

### 3. Validation 
Financial time series break most of the assumptions behind standard k-fold
cross-validation. This project uses:
- **Walk-forward (expanding window) validation**: Train only on data strictly
  before the test period, roll forward, repeat. No shuffling, ever.
- **Strict lagging of all features**: Every feature at time `t` uses only
  information available at or before `t`, to eliminate look-ahead bias (the
  single most common error identified in quant research post-mortems).
- **QLIKE and RMSE loss**: QLIKE is the standard loss function in the
  volatility forecasting literature (Patton, 2011) because, unlike plain RMSE,
  it is robust to the choice of (imperfect) volatility proxy used as ground
  truth.
- **A naive baseline** (yesterday's realized vol = tomorrow's forecast) is
  always included as a sanity floor. If a "sophisticated" model can't beat
  this, that itself is a finding worth reporting honestly.


## Repo structure

```
vol_forecast_project/
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py      # fetch & cache market data
│   ├── features.py         # leak-free feature engineering
│   ├── validation.py       # walk-forward CV framework
│   ├── models/
│   │   ├── baseline_garch.py
│   │   ├── ml_xgboost.py
│   │   └── dl_lstm.py
│   ├── evaluate.py         # RMSE, QLIKE, directional accuracy
│   └── main.py              # orchestrates the full pipeline
├── gcp/
│   ├── upload_to_bigquery.py
│   ├── vertex_ai_train.py
│   └── README_gcp.md
└── results/                 # created when you run main.py                
```