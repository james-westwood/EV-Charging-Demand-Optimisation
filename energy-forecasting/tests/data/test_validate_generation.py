"""Tests for src/data/validators/generation_mix.py."""
import pandas as pd
import pytest

from src.data.validators.generation_mix import validate_generation_mix
from src.data.validators.exceptions import ValidationError

_FUEL_COLUMNS = ["gas", "coal", "nuclear", "wind", "hydro", "imports", "biomass", "other", "solar"]


def _make_df(n: int = 5, start: str = "2024-01-01 00:00", **overrides) -> pd.DataFrame:
    """Build a valid generation mix DataFrame with n rows.
    
    To be valid according to the 'row max' rule, one fuel must be dominant,
    or we must provide a 'total' column.
    """
    periods = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    # One dominant fuel to pass the 'row max' rule (sum <= 1.05 * max)
    data = {
        "settlement_period": periods,
        "gas": [100.0] * n,
        "coal": [0.0] * n,
        "nuclear": [2.0] * n,
        "wind": [1.0] * n,
        "hydro": [0.0] * n,
        "imports": [0.0] * n,
        "biomass": [1.0] * n,
        "other": [0.0] * n,
        "solar": [0.5] * n,
    }
    # sum = 104.5, max = 100.0. sum is 1.045 * max, which is within 1.05 * max.
    
    df = pd.DataFrame(data)
    for col, values in overrides.items():
        df[col] = values
    return df


class TestValidCases:
    def test_valid_with_dominant_fuel_passes(self):
        df = _make_df(n=5)
        validate_generation_mix(df)  # should not raise

    def test_valid_with_total_column_passes(self):
        # Even if no dominant fuel, if total is present it should pass
        periods = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")
        df = pd.DataFrame({
            "settlement_period": periods,
            "gas": [50.0] * 5,
            "wind": [50.0] * 5,
            "total": [100.0] * 5
        })
        validate_generation_mix(df)

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame(columns=["settlement_period"] + _FUEL_COLUMNS)
        validate_generation_mix(df)


class TestFuelOutOfRange:
    def test_negative_fuel_raises(self):
        df = _make_df(n=5)
        df.at[2, "wind"] = -5.0
        with pytest.raises(ValidationError) as exc_info:
            validate_generation_mix(df)
        assert exc_info.value.field == "wind"
        assert "wind" in str(exc_info.value)

    def test_negative_gas_raises(self):
        df = _make_df(n=5)
        df.at[0, "gas"] = -0.1
        with pytest.raises(ValidationError) as exc_info:
            validate_generation_mix(df)
        assert exc_info.value.field == "gas"


class TestRowSumConsistency:
    def test_sum_exceeds_row_max_raises(self):
        # sum = 100 (gas) + 10 (coal) = 110. max = 100. 110 > 1.05 * 100.
        df = _make_df(n=5, gas=[100.0]*5, coal=[10.0]*5)
        with pytest.raises(ValidationError) as exc_info:
            validate_generation_mix(df)
        assert exc_info.value.field == "multiple"

    def test_sum_outside_total_range_raises(self):
        periods = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")
        df = pd.DataFrame({
            "settlement_period": periods,
            "gas": [60.0] * 5,
            "total": [100.0] * 5  # sum is 60, total is 100. 60 < 95.
        })
        with pytest.raises(ValidationError) as exc_info:
            validate_generation_mix(df)
        assert exc_info.value.field == "total"


class TestMissingTimestamps:
    def test_null_settlement_period_raises(self):
        df = _make_df(n=5)
        df.at[1, "settlement_period"] = None
        with pytest.raises(ValidationError) as exc_info:
            validate_generation_mix(df)
        assert exc_info.value.field == "settlement_period"

    def test_duplicate_timestamps_raise(self):
        df = _make_df(n=5)
        df.at[3, "settlement_period"] = df.at[0, "settlement_period"]
        with pytest.raises(ValidationError) as exc_info:
            validate_generation_mix(df)
        assert exc_info.value.field == "settlement_period"

    def test_irregular_interval_raises(self):
        df = _make_df(n=3)
        # Change 3rd timestamp to be 1 hour later instead of 30 min
        df.at[2, "settlement_period"] = df.at[1, "settlement_period"] + pd.Timedelta(hours=1)
        with pytest.raises(ValidationError) as exc_info:
            validate_generation_mix(df)
        assert exc_info.value.field == "settlement_period"
