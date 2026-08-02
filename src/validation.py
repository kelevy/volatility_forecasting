"""
Walk-forward (expanding window) cross-validation for time series.

Standard k-fold CV shuffles rows, which is invalid for time series: it lets
information from the future leak into training via temporal autocorrelation,
and it violates the causal structure the model is supposed to learn. This
module implements the expanding-window scheme used in quantitative finance
research instead.

Scheme:
  fold 1: train [0 : t1)          test [t1 : t1+step)
  fold 2: train [0 : t1+step)     test [t1+step : t1+2*step)
  ...
i.e. the training window only ever grows forward in time; test windows never
overlap with or precede their training data.
"""

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold_number: int


def walk_forward_splits(
    n_rows: int,
    n_splits: int = 5,
    min_train_size: float = 0.5,
) -> Iterator[Fold]:
    """
    Yields Fold objects with strictly non-overlapping, forward-only train/test
    index arrays over a dataset of length n_rows.

    min_train_size: fraction of the data used for the first training window,
    before any expansion.
    """
    first_train_end = int(n_rows * min_train_size)
    remaining = n_rows - first_train_end
    step = remaining // n_splits

    if step <= 0:
        raise ValueError("Not enough data for the requested n_splits.")

    for i in range(n_splits):
        train_end = first_train_end + i * step
        test_start = train_end
        test_end = min(train_end + step, n_rows)

        if test_start >= test_end:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        # Defensive assertion: no overlap, no leakage, strictly forward.
        assert train_idx.max() < test_idx.min(), "Leakage: train overlaps test!"

        yield Fold(train_idx=train_idx, test_idx=test_idx, fold_number=i + 1)


def split_xy(df: pd.DataFrame, feature_cols, target_col: str, fold: Fold):
    X_train = df.iloc[fold.train_idx][feature_cols]
    y_train = df.iloc[fold.train_idx][target_col]
    X_test = df.iloc[fold.test_idx][feature_cols]
    y_test = df.iloc[fold.test_idx][target_col]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Sanity check with synthetic data -- verifies no leakage across folds.
    n = 1000
    for fold in walk_forward_splits(n, n_splits=5):
        print(
            f"Fold {fold.fold_number}: "
            f"train=[0:{fold.train_idx.max()+1}] ({len(fold.train_idx)} rows)  "
            f"test=[{fold.test_idx.min()}:{fold.test_idx.max()+1}] ({len(fold.test_idx)} rows)"
        )