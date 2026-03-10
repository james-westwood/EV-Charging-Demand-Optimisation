"""Tests for src/data/validators/carbon_intensity.py."""
import pandas as pd
import pytest

from src.data.validators.carbon_intensity import validate_carbon_intensity
from src.data.validators.exceptions import ValidationError


def _make_df(n: int = 5, start: str = "2024-01-01 00:00", **overrides) -> pd.DataFrame:
    """Build a valid carbon intensity DataFrame with n rows."""
    periods = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "settlement_period": periods,
            "intensity_actual": [200] * n,
            "intensity_forecast": [210] * n,
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


class TestValidCases:
    def test_valid_dataframe_passes(self):
        df = _make_df(n=10)
        validate_carbon_intensity(df)  # should not raise

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame(columns=["settlement_period", "intensity_actual", "intensity_forecast"])
        validate_carbon_intensity(df)

    def test_single_row_passes(self):
        df = _make_df(n=1)
        validate_carbon_intensity(df)

    def test_boundary_intensity_zero(self):
        df = _make_df(intensity_actual=[0] * 5)
        validate_carbon_intensity(df)

    def test_boundary_intensity_800(self):
        df = _make_df(intensity_actual=[800] * 5)
        validate_carbon_intensity(df)

    def test_null_intensity_actual_is_allowed(self):
        """Null intensity values should be skipped (API sometimes returns None)."""
        df = _make_df(n=5)
        df.at[2, "intensity_actual"] = None
        validate_carbon_intensity(df)


class TestIntensityOutOfRange:
    def test_intensity_actual_above_max_raises(self):
        df = _make_df(intensity_actual=[999] * 5)
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert exc_info.value.field == "intensity_actual"

    def test_intensity_actual_negative_raises(self):
        df = _make_df(intensity_actual=[-1] * 5)
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert exc_info.value.field == "intensity_actual"

    def test_intensity_forecast_above_max_raises(self):
        df = _make_df(intensity_forecast=[801] * 5)
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert exc_info.value.field == "intensity_forecast"

    def test_single_bad_row_raises(self):
        """Only one row has an out-of-range value — should still raise."""
        df = _make_df(n=5)
        df.at[3, "intensity_actual"] = 999
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert exc_info.value.field == "intensity_actual"


class TestNullSettlementPeriod:
    def test_null_settlement_period_raises(self):
        df = _make_df(n=5)
        df.at[1, "settlement_period"] = None
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert exc_info.value.field == "settlement_period"


class TestDuplicateTimestamps:
    def test_duplicate_timestamps_raise(self):
        df = _make_df(n=5)
        df.at[3, "settlement_period"] = df.at[0, "settlement_period"]
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert exc_info.value.field == "settlement_period"


class TestIntervalCheck:
    def test_irregular_interval_raises(self):
        """A 1-hour gap between rows should fail the 30-min interval check."""
        periods = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC").tolist()
        # Insert a 60-min gap between index 1 and 2
        periods[2] = periods[1] + pd.Timedelta(hours=1)
        df = pd.DataFrame(
            {
                "settlement_period": periods,
                "intensity_actual": [200, 200, 200],
                "intensity_forecast": [210, 210, 210],
            }
        )
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert exc_info.value.field == "settlement_period"

    def test_error_message_mentions_field(self):
        df = _make_df(intensity_actual=[999] * 5)
        with pytest.raises(ValidationError) as exc_info:
            validate_carbon_intensity(df)
        assert "intensity_actual" in str(exc_info.value)
