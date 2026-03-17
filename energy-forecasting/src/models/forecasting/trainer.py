"""Trains one LightGBM model for a given quantile (alpha). Will be called it three times 
— once each for P10, P50, P90. It returns the
  trained model plus out-of-fold predictions covering the full dataset.
"""

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMRegressor

from src.models.forecasting.cv import time_series_cv_split
from src.models.forecasting.metrics import pinball_loss


def train_quantile_lgbm(
      X: pd.DataFrame,
      y: pd.Series,
      alpha: float,
      n_splits: int = 5,
      gap: int = 48,
    ) -> tuple[Booster, np.ndarray]:
    """- X — feature DataFrame from your feature pipeline
    - y — target Series (e.g. carbon_intensity)     
    - alpha — 0.1, 0.5, or 0.9
    - Returns (fitted_model, oof_predictions)
    
    """
    params = {
      "objective": "quantile",
      "alpha": alpha,          # ← this is what changes between P10/P50/P90
      "metric": "quantile",
      "n_estimators": 300,
      "learning_rate": 0.05,
      "num_leaves": 31,
      "verbose": -1,           # silence training output
    }
    with mlflow.start_run(run_name=f"lgbm_quantile_alpha_{alpha}"):
        mlflow.log_params(params)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("gap", gap)
        mlflow.log_param("training_rows", len(X))
        
        # cv time series splitting
        cv = time_series_cv_split(len(X), n_splits=n_splits, gap=gap)
        
        # initialise oof
        oof_preds = np.full(len(X), np.nan)
        
        for train_idx, val_idx in cv:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
        
            fold_model = LGBMRegressor(**params)  # type: ignore[invalid-argument-type]
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx] = fold_model.predict(X_val)
                
        final_model = LGBMRegressor(**params)  # type: ignore[invalid-argument-type]  # 1. create untrained model                                                                              
        final_model.fit(X, y)                  # 2. train it on ALL data
        
        # Log oof pinball loss ignoring nans
        mask = ~np.isnan(oof_preds)
        loss = pinball_loss(alpha, oof_preds[mask], y.values[mask])
        mlflow.log_metric("oof_pinball_loss", loss)
        mlflow.lightgbm.log_model(final_model, f"lgbm_p{int(alpha*100)}")
        
    return final_model, oof_preds          # 3. return the now-trained model  

if __name__ == "__main__":
    X = pd.DataFrame(np.random.randn(1000, 5), columns=list("ABCDE"))  # type: ignore[invalid-argument-type]
    y = pd.Series(np.random.randn(1000))
    print(train_quantile_lgbm(X, y, alpha=0.5))