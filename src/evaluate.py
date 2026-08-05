"""
Metrics for comparing volatility forecasts.

- RMSE: standard, intuitive, but sensitive to the noisiness of the realized
  volatility proxy used as ground truth.
- QLIKE: the standard loss function in the academic volatility-forecasting
  literature (Patton, 2011, Journal of Econometrics). More robust than RMSE
  to noise in the volatility proxy because it penalizes relative, not
  absolute, error -- this matters because "true" volatility is unobservable
  and realized volatility is itself a noisy estimate of it.
- Directional accuracy: did the model correctly predict whether volatility
  would rise or fall vs. the previous period? Relevant for practical
  positioning decisions, not just point-forecast accuracy.
"""

import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def qlike(y_true, y_pred, eps: float = 1e-8):
    """
    QLIKE loss: mean( y_true/y_pred - log(y_true/y_pred) - 1 )
    Requires strictly positive values (volatility forecasts should be).
    """
    y_true = np.clip(np.asarray(y_true), eps, None)
    y_pred = np.clip(np.asarray(y_pred), eps, None)
    ratio = y_true / y_pred
    return np.mean(ratio - np.log(ratio) - 1)


def directional_accuracy(y_true, y_pred, y_prev):
    """
    Fraction of times the model correctly predicts whether volatility rises
    or falls relative to the previous period's realized vol (y_prev).
    """
    y_true, y_pred, y_prev = map(np.asarray, (y_true, y_pred, y_prev))
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return np.mean(true_dir == pred_dir)


def evaluate_all(y_true, y_pred, y_prev):
    return {
        "RMSE": rmse(y_true, y_pred),
        "QLIKE": qlike(y_true, y_pred),
        "DirectionalAccuracy": directional_accuracy(y_true, y_pred, y_prev),
    }


def results_table(results: dict):
    """results: {model_name: {metric: value}} -> tidy DataFrame."""
    return pd.DataFrame(results).T


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0.1, 0.3, 200)
    y_pred = y_true + rng.normal(0, 0.02, 200)
    y_prev = np.roll(y_true, 1)

    print(evaluate_all(y_true, y_pred, y_prev))