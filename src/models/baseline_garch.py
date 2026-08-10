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
    """
    results = []
    returns_pct = log_returns * 100

    for fold in folds:
        train_returns = returns_pct.iloc[fold.train_idx]
        model = GarchBaseline(horizon=horizon).fit(train_returns)
        forecast = model.forecast_annualized_vol()
        results.append((fold.fold_number, forecast))

    return results


if __name__ == "__main__":
    # Smoke test with synthetic returns (requires `arch` installed)
    rng = np.random.default_rng(0)
    n = 1000
    synthetic_returns = pd.Series(rng.normal(0, 1.0, n))  # already in "percent" scale

    model = GarchBaseline(horizon=5).fit(synthetic_returns)
    print("Forecast (annualized vol):", model.forecast_annualized_vol())