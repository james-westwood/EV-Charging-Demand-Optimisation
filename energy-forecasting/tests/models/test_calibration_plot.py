import numpy as np
import pytest

from src.models.forecasting.calibration_plot import calculate_calibration


def test_calculate_calibration_perfect():
    """P10/P50/P90 predictions that match exact quantiles give expected observed frequencies."""
    rng = np.random.default_rng(42)
    actuals = rng.uniform(0, 100, size=10_000)
    # Shift predictions so exactly α of actuals fall below
    p10_preds = np.quantile(actuals, 0.10) * np.ones(len(actuals))
    p50_preds = np.quantile(actuals, 0.50) * np.ones(len(actuals))
    p90_preds = np.quantile(actuals, 0.90) * np.ones(len(actuals))

    result = calculate_calibration(actuals, p10_preds, p50_preds, p90_preds)

    assert result["p10_claimed"] == pytest.approx(0.1)
    assert result["p50_claimed"] == pytest.approx(0.5)
    assert result["p90_claimed"] == pytest.approx(0.9)
    assert result["p10_observed"] == pytest.approx(0.10, abs=0.02)
    assert result["p50_observed"] == pytest.approx(0.50, abs=0.02)
    assert result["p90_observed"] == pytest.approx(0.90, abs=0.02)


def test_calculate_calibration_ignores_nan_rows():
    """NaN values in OOF predictions (from CV gaps) are excluded from the calculation."""
    actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Introduce a NaN in first row; valid rows have actuals (2,3,4,5) < p50_preds (10)
    p10_preds = np.array([np.nan, 0.5, 0.5, 0.5, 0.5])
    p50_preds = np.array([np.nan, 10.0, 10.0, 10.0, 10.0])
    p90_preds = np.array([np.nan, 20.0, 20.0, 20.0, 20.0])

    result = calculate_calibration(actuals, p10_preds, p50_preds, p90_preds)

    # Only 4 valid rows; p50 (10.0) > all 4 valid actuals → observed = 1.0
    assert result["p50_observed"] == pytest.approx(1.0)
    # p10 (0.5) < all 4 valid actuals → observed = 0.0
    assert result["p10_observed"] == pytest.approx(0.0)
    # p90 (20.0) > all 4 valid actuals → observed = 1.0
    assert result["p90_observed"] == pytest.approx(1.0)


def test_calculate_calibration_all_nan_returns_nan():
    """When every row is NaN, observed frequencies are returned as NaN."""
    n = 10
    nan_arr = np.full(n, np.nan)
    actuals = np.ones(n)

    result = calculate_calibration(actuals, nan_arr, nan_arr, nan_arr)

    assert np.isnan(result["p10_observed"])
    assert np.isnan(result["p50_observed"])
    assert np.isnan(result["p90_observed"])


def test_calculate_calibration_returns_correct_keys():
    """Result dict contains all six expected keys."""
    actuals = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.5, 2.5, 3.5])

    result = calculate_calibration(actuals, preds, preds, preds)

    expected_keys = {"p10_claimed", "p10_observed", "p50_claimed", "p50_observed", "p90_claimed", "p90_observed"}
    assert set(result.keys()) == expected_keys


def test_calculate_calibration_nan_in_actuals_excluded():
    """NaN in actuals is excluded from the calculation."""
    actuals = np.array([np.nan, 2.0, 3.0, 4.0])
    preds = np.array([1.0, 5.0, 5.0, 5.0])

    result = calculate_calibration(actuals, preds, preds, preds)

    # 3 valid rows: actuals (2,3,4) all < preds (5) → observed = 1.0
    assert result["p50_observed"] == pytest.approx(1.0)
