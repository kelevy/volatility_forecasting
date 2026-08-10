"""
Gradient-boosted trees on the engineered feature set (see features.py).
This is the ML model in the comparison -- more flexible than GARCH, but
still using hand-engineered features rather than learning directly from the
raw sequence like the LSTM model.
"""

import numpy as np
import pandas as pd


class XGBVolModel:
    def __init__(self, **xgb_params):
        default_params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        default_params.update(xgb_params)
        self.params = default_params
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        from xgboost import XGBRegressor

        self.model = XGBRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .fit() before predict().")
        preds = self.model.predict(X_test)
        # volatility forecasts should be non-negative
        return np.clip(preds, a_min=1e-6, a_max=None)

    def feature_importances(self, feature_names) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call .fit() before requesting importances.")
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)


def run_xgb_walk_forward(feature_df: pd.DataFrame, feature_cols, target_col, folds):
    """
    Runs XGBoost across walk-forward folds. Returns predictions/actuals
    concatenated across all test folds, ready for evaluate.py.
    """
    from validation import split_xy

    all_preds, all_actuals, all_prev = [], [], []

    for fold in folds:
        X_train, y_train, X_test, y_test = split_xy(feature_df, feature_cols, target_col, fold)

        model = XGBVolModel().fit(X_train, y_train)
        preds = model.predict(X_test)

        # y_prev: the trailing realized vol feature already in the frame,
        # used only for directional-accuracy scoring, not as a prediction.
        y_prev = feature_df.iloc[fold.test_idx]["realized_vol_lag1"].values

        all_preds.append(preds)
        all_actuals.append(y_test.values)
        all_prev.append(y_prev)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_actuals),
        np.concatenate(all_prev),
    )


if __name__ == "__main__":
    # Smoke test with synthetic data 
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(0.2 + 0.05 * X["f0"] + rng.normal(0, 0.01, n))

    model = XGBVolModel().fit(X.iloc[:400], y.iloc[:400])
    preds = model.predict(X.iloc[400:])
    print("Sample predictions:", preds[:5])
    print("Feature importances:\n", model.feature_importances(X.columns))