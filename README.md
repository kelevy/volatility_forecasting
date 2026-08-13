# Volatility Forecasting: GARCH vs. Gradient Boosting vs. LSTM

A research-style comparison of three approaches to forecasting realized volatility
on the S&P 500: a classical statistical baseline, a modern machine learning model, and a
deep learning model, evaluated under a walk-forward (no look-ahead) validation
scheme with financially meaningful loss functions.

## Motivation

Volatility forecasting is a core problem in quantitative finance, used in
options pricing, risk management, and position sizing. This project asks:
**does adding model complexity (ML, DL) actually improve volatility
forecasts over a classical econometric baseline?**

## Methodology

### 1. Data
- S&P 500 (`^GSPC`) daily OHLC data via `yfinance`, ~20 years of history (2005–present).
- Target: realized volatility, computed as the rolling standard deviation of
  log returns over the next `h` trading days (forecast horizon), annualized.

### 2. Models compared

The core research question is a **three-way comparison**: classical econometric
vs. ML vs. DL. A fourth, deliberately trivial **naive baseline** is included
alongside them as a sanity floor.

| Model | Type | Role |
|---|---|---|
| Naive | Persistence rule | **Sanity floor**: Yesterday's realized vol = tomorrow's forecast. Not part of the core comparison. |
| GARCH(1,1) | Classical econometric | **Baseline**: The industry-standard model for volatility clustering. Forecasts daily on an expanding window. |
| XGBoost | Gradient-boosted trees | **ML model**: Engineered features (lagged returns, rolling volatility, momentum). |
| LSTM |  Recurrent neural network | **DL model**: Learns directly from raw return sequences rather than hand-engineered features. |

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
- **Consistent directional-accuracy scoring**: every model's "previous value"
  reference for directional accuracy is the same (`realized_vol_lag1`), so the
  `DirectionalAccuracy` column is directly comparable across all four rows.

## Results

5-fold walk-forward validation, S&P 500, 2005–2026 (~5,400 trading days).

| Model | RMSE | QLIKE | Directional Accuracy |
|---|---|---|---|
| Naive | 0.106458 | 0.220172 | 0.000000 |
| GARCH(1,1) | 0.098008 | 0.134386 | 0.643056 |
| **XGBoost** | **0.095197** | **0.126316** | **0.690741** |
| LSTM | 0.130194 | 0.230464 | 0.603704 |

**Summary**: XGBoost achieved the best point-forecast accuracy and directional
accuracy of all models tested. GARCH outperformed the naive baseline on every
metric, confirming it captures real structure before being compared against
ML/DL. LSTM trailed both GARCH and XGBoost despite normalized inputs and
seeded, stable training, consistent with the general finding that deep
learning typically needs substantially more data than a single ~20-year daily
series (~5,400 observations) provides to reliably outperform well-specified
classical and tree-based methods.

XGBoost's top features were dominated by rolling standard deviations of
returns (`ret_roll_std_10`, `ret_roll_std_21`, `ret_roll_std_5`), followed by
`realized_vol_lag1`, consistent with volatility clustering: recent return
dispersion across multiple trailing windows is more informative than any
single lagged volatility estimate alone.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This downloads S&P 500 data (cached locally after the first run), builds the
leak-free feature set, runs all four models through walk-forward validation,
and saves `results/results.csv` and `results/forecasts.csv`.


## Notebooks

- **`notebooks/01_exploratory_analysis.ipynb`**: Standalone notebook, explores the raw
  data (return distribution, volatility clustering, autocorrelation of squared
  returns, feature correlations). Can be run independently of `main.py`.
- **`notebooks/02_results_analysis.ipynb`**: Depends on `main.py` having
  already run, since it loads `results/results.csv` and `results/forecasts.csv`.
  Produces the model comparison charts, error analysis, and the Conclusions
  section summarized above.

## Repo structure

```
vol_forecast_project/
├── README.md
├── requirements.txt
├── main.py                 # runs the full pipeline
├── src/
│   ├── data_loader.py      # fetch & cache market data
│   ├── features.py         # leak-free feature engineering
│   ├── validation.py       # walk-forward CV framework
│   ├── evaluate.py         # RMSE, QLIKE, directional accuracy
│   └── models/
│       ├── baseline_garch.py
│       ├── ml_xgboost.py
│       └── dl_lstm.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_results_analysis.ipynb
├── data/                   # created on first run (gitignored)
└── results/                # created when you run main.py (gitignored)
```
