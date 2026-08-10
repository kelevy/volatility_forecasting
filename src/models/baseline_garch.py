"""
GARCH(1,1) baseline -- the industry-standard classical model for volatility
clustering. This is the benchmark the ML/DL models need to beat.

Uses the `arch` package (Kevin Sheppard's ARCH toolbox), the standard tool
for GARCH modelling in Python.
"""

import numpy as np
import pandas as pd


class GarchBaseline:
    """
    Fits GARCH(1,1) on daily log returns and forecasts realized volatility
    over the same horizon used elsewhere in the project (vol_window days),
    refitting at each walk-forward fold on that fold's training data only.
    """

    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self._fitted_result = None

    def fit(self, returns_pct: pd.Series):
        """
        returns_pct: log returns *in percent* (arch package convention --
        fitting on raw decimal returns can cause numerical instability).
        """
        from arch import arch_model

        am = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
        self._fitted_result = am.fit(disp="off")
        return self

    def forecast_annualized_vol(self) -> float:
        """
        Returns a single annualized volatility forecast for the next
        `self.horizon` days, aggregating the per-day variance forecasts.
        """
        if self._fitted_result is None:
            raise RuntimeError("Call .fit() before forecasting.")

        fc = self._fitted_result.forecast(horizon=self.horizon, reindex=False)
        # variance forecasts are in percent^2 units (since we fit on pct returns)
        daily_var_pct2 = fc.variance.values[-1]  # shape (horizon,)
        total_var_pct2 = daily_var_pct2.sum()
        # convert percent^2 back to decimal, annualize
        annualized_vol = np.sqrt(total_var_pct2) / 100 * np.sqrt(252 / self.horizon)
        return float(annualized_vol)


def run_garch_walk_forward(log_returns: pd.Series, folds, horizon: int = 5) -> list:
    """
    Runs the GARCH baseline across walk-forward folds. Refits at each fold
    using only that fold's training window (no peeking ahead), then produces
    ONE forecast per fold for the first test-period date.

    Returns a list of (fold_number, forecast) tuples.

    NOTE: kept for reference / simple use cases. For the actual project
    evaluation, use `forecast_daily_expanding` below instead -- broadcasting
    one static forecast across an entire ~78-day test window understates
    GARCH's real forecasting ability.
    """
    results = []
    returns_pct = log_returns * 100

    for fold in folds:
        train_returns = returns_pct.iloc[fold.train_idx]
        model = GarchBaseline(horizon=horizon).fit(train_returns)
        forecast = model.forecast_annualized_vol()
        results.append((fold.fold_number, forecast))

    return results


def forecast_daily_expanding(
    log_returns: pd.Series,
    folds,
    horizon: int = 5,
    refit_every: int = 5,
) -> np.ndarray:
    """
    For each day in each fold's test window, fits GARCH on all returns
    strictly before that day (expanding window) and produces a fresh
    horizon-day-ahead annualized volatility forecast -- rather than
    broadcasting one static forecast across the whole test window.

    refit_every: re-estimate GARCH parameters every N days rather than
    every single day (refitting daily is the "purest" approach but is slow
    and the parameter estimates barely move day-to-day; refitting weekly is
    a standard, defensible compromise between rigor and runtime). The
    conditional variance recursion itself still uses every new return as it
    becomes available -- only the *parameter re-estimation* is throttled.

    Returns predictions in the same order/concatenation as
    validation.walk_forward_splits folds (matches feature_df row order
    within each fold's test_idx).
    """
    returns_pct = log_returns * 100
    all_preds = []

    for fold in folds:
        model = None
        for i, test_pos in enumerate(fold.test_idx):
            # Expanding window: all data strictly before this test day.
            history_end = test_pos  # exclusive upper bound
            train_returns = returns_pct.iloc[:history_end]

            if model is None or i % refit_every == 0:
                model = GarchBaseline(horizon=horizon).fit(train_returns)
            else:
                # Reuse existing fitted parameters, just extend the
                # conditional variance recursion with newly available data.
                from arch import arch_model

                am = arch_model(train_returns, vol="Garch", p=1, q=1, dist="normal")
                model._fitted_result = am.fix(model._fitted_result.params)

            forecast = model.forecast_annualized_vol()
            all_preds.append(forecast)

    return np.array(all_preds)


if __name__ == "__main__":
    # Smoke test with synthetic returns 
    rng = np.random.default_rng(0)
    n = 1000
    synthetic_returns = pd.Series(rng.normal(0, 1.0, n))  # already in "percent" scale

    model = GarchBaseline(horizon=5).fit(synthetic_returns)
    print("Forecast (annualized vol):", model.forecast_annualized_vol())