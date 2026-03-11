import pytest 

import numpy as np

from src.models.forecasting.cv import time_series_cv_split  


def test_no_leakage():
      splits = time_series_cv_split(1000, n_splits=5, gap=48)
      for train, val in splits:
          assert val.min() > train.max()
          assert len(np.intersect1d(train, val)) == 0

def test_returns_5_folds():                                                                                                                     
      splits = time_series_cv_split(1000, n_splits=5, gap=48)                                                                                     
      assert len(splits) == 5 
      
def test_gap_is_respected():                                                                                                                    
      splits = time_series_cv_split(1000, n_splits=5, gap=48)
      for train, val in splits:                                                                                                                   
          assert val.min() - train.max() > 48  # gap enforced between train end and val start



if __name__ == "__main__":
    # 5 folds, gap=48 periods (1 day),
    test_no_leakage()
    
    