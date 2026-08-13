"""
LSTM sequence model -- learns directly from a rolling window of past returns.

Uses PyTorch. Kept deliberately small (single LSTM layer, small hidden size)
since this is a single-asset, moderate-data-size problem -- a huge network
would overfit.
"""

import numpy as np
import pandas as pd


def build_sequences(log_returns: pd.Series, target: pd.Series, seq_len: int = 21):
    """
    Builds (X, y) where each X[i] is a window of `seq_len` trailing log
    returns [t-seq_len ... t-1], and y[i] is target[t].

    Strictly no look-ahead: X[i] only uses returns up to and including t-1.
    """
    returns = log_returns.values
    targets = target.values
    n = len(returns)

    X, y, idx = [], [], []
    for t in range(seq_len, n):
        window = returns[t - seq_len : t]
        if np.isnan(window).any() or np.isnan(targets[t]):
            continue
        X.append(window)
        y.append(targets[t])
        idx.append(t)

    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
    return X, y, np.array(idx)


class LSTMVolModel:
    def __init__(self, seq_len: int = 21, hidden_size: int = 16, epochs: int = 100, lr: float = 1e-3):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.net = None
        self._x_mean = None
        self._x_std = None

    def _normalize_fit(self, X_train: np.ndarray):
        """
        Fits normalization stats on TRAINING data only, then
        returns the normalized training set. 
        """
        self._x_mean = X_train.mean()
        self._x_std = X_train.std() + 1e-8
        return (X_train - self._x_mean) / self._x_std

    def _normalize_apply(self, X: np.ndarray) -> np.ndarray:
        if self._x_mean is None:
            raise RuntimeError("Call .fit() before predict() -- no normalization stats yet.")
        return (X - self._x_mean) / self._x_std

    def _build_net(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())  # positivity

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                out = self.head(h_n[-1])
                return out.squeeze(-1)

        return Net(self.hidden_size)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        import torch
        import torch.nn as nn

        X_train_norm = self._normalize_fit(X_train)

        self.net = self._build_net()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X_train_norm)
        y_t = torch.tensor(y_train)

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = self.net(X_t)
            loss = loss_fn(preds, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            optimizer.step()

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        import torch

        if self.net is None:
            raise RuntimeError("Call .fit() before predict().")

        X_test_norm = self._normalize_apply(X_test)

        self.net.eval()
        with torch.no_grad():
            preds = self.net(torch.tensor(X_test_norm)).numpy()
        return preds


def run_lstm_walk_forward(log_returns: pd.Series, target: pd.Series, prev_series: pd.Series,
                           folds, seq_len: int = 21):
    """
    Runs the LSTM across walk-forward folds. 
    """
    X_all, y_all, seq_idx = build_sequences(log_returns, target, seq_len=seq_len)

    all_preds, all_actuals, all_prev = [], [], []
    prev_vals = prev_series.values

    for fold in folds:
        train_mask = np.isin(seq_idx, fold.train_idx)
        test_mask = np.isin(seq_idx, fold.test_idx)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = LSTMVolModel().fit(X_all[train_mask], y_all[train_mask])
        preds = model.predict(X_all[test_mask])

        all_preds.append(preds)
        all_actuals.append(y_all[test_mask])
        all_prev.append(prev_vals[seq_idx[test_mask]])

    return (
        np.concatenate(all_preds),
        np.concatenate(all_actuals),
        np.concatenate(all_prev),
    )


if __name__ == "__main__":
    # Smoke test with synthetic data 
    rng = np.random.default_rng(0)
    n, seq_len = 500, 21
    returns = pd.Series(rng.normal(0, 0.01, n))
    target = pd.Series(np.abs(rng.normal(0.2, 0.03, n)))
    prev_series = target.shift(1).fillna(target.iloc[0])  

    X, y, idx = build_sequences(returns, target, seq_len=seq_len)
    print("Sequence tensor shape:", X.shape)

    model = LSTMVolModel(epochs=5).fit(X[:400], y[:400])
    preds = model.predict(X[400:410])
    print("Sample predictions:", preds)