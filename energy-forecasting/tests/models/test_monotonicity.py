from src.models.forecasting.monotonicity import check_quantile_monotonicity
import numpy as np



def test_quantile_monotonicity():
# Test with 5 p10 values > than p50
    p10 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p50 = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    p90 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    assert check_quantile_monotonicity(p10, p50, p90) == {'violation_count': 5, 'violation_pct': 0.5}