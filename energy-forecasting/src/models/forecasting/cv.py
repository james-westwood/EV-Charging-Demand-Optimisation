"""TimeSeriesSplitter for cross-validation."""
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


def time_series_cv_split(n: int, n_splits: int = 5, gap: int = 48) -> list[tuple[np.ndarray, np.ndarray]]:
      """Return list of (train_idx, val_idx) tuples with strict temporal ordering.

      Args:
          n:        Total number of rows in the dataset.
          n_splits: Number of CV folds (default 5).
          gap:      Periods to skip between train and val (default 48 = 1 day).

      Returns:
          List of (train_indices, val_indices) tuples. Within each tuple,
          all val indices are strictly greater than all train indices.
      """
      splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
      indices = np.arange(n)
      return [(train, val) for train, val in splitter.split(indices)]


if __name__ == "__main__":
    splits = time_series_cv_split(1000, n_splits=5, gap=48)                                                                                 
                                                                                                                                                  
    for i, (train, val) in enumerate(splits):                                                                                                       
      print(f'Fold {i+1}:  train={len(train)} rows  [{train[0]}..{train[-1]}]  |  gap  |  val={len(val)} rows  [{val[0]}..{val[-1]}]')
    
    
    