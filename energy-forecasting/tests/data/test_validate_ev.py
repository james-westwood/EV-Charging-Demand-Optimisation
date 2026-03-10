"""Tests for src/data/validators/ev_sessions.py."""
import pandas as pd
import pytest

from src.data.validators.exceptions import ValidationError
from src.data.validators.ev_sessions import validate_ev_sessions


def _make_df(
    n: int = 3,
    energy_kwh: float = 20.0,
    duration: float = 60.0,
    offset_minutes: int = 90,
) -> pd.DataFrame:
    """Build a valid EV sessions DataFrame with n rows."""
    base = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    rows = []
    for i in range(n):
        arrival = base + pd.Timedelta(hours=i * 2)
        departure = arrival + pd.Timedelta(minutes=offset_minutes)
        rows.append(
            {
                "station_id": f"station_{i}",
                "arrival_time": arrival,
                "departure_time": departure,
                "energy_kwh": energy_kwh,
                "duration": duration,
            }
        )
    return pd.DataFrame(rows)


class TestValidCases:
    def test_valid_dataframe_passes(self):
        df = _make_df()
        validate_ev_sessions(df)  # should not raise

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame(
            columns=["station_id", "arrival_time", "departure_time", "energy_kwh", "duration"]
        )
        validate_ev_sessions(df)

    def test_null_energy_kwh_is_skipped(self):
        df = _make_df()
        df.at[0, "energy_kwh"] = None
        validate_ev_sessions(df)

    def test_null_duration_is_skipped(self):
        df = _make_df()
        df.at[0, "duration"] = None
        validate_ev_sessions(df)


class TestEnergyKwhValidation:
    def test_zero_energy_raises(self):
        df = _make_df(energy_kwh=0.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "energy_kwh"

    def test_negative_energy_raises(self):
        df = _make_df(energy_kwh=-5.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "energy_kwh"

    def test_single_bad_energy_row_raises(self):
        df = _make_df()
        df.at[1, "energy_kwh"] = 0.0
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "energy_kwh"

    def test_error_message_mentions_field(self):
        df = _make_df(energy_kwh=-1.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert "energy_kwh" in str(exc_info.value)


class TestDepartureArrivalValidation:
    def test_departure_before_arrival_raises(self):
        df = _make_df()
        # Set departure before arrival for one row
        df.at[0, "departure_time"] = df.at[0, "arrival_time"] - pd.Timedelta(minutes=10)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "departure_time"

    def test_departure_equal_arrival_raises(self):
        df = _make_df()
        df.at[0, "departure_time"] = df.at[0, "arrival_time"]
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "departure_time"

    def test_all_departures_before_arrivals_raises(self):
        df = _make_df(offset_minutes=-30)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "departure_time"

    def test_error_message_mentions_field(self):
        df = _make_df()
        df.at[0, "departure_time"] = df.at[0, "arrival_time"] - pd.Timedelta(minutes=1)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert "departure_time" in str(exc_info.value)


class TestDurationValidation:
    def test_zero_duration_raises(self):
        df = _make_df(duration=0.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "duration"

    def test_negative_duration_raises(self):
        df = _make_df(duration=-10.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "duration"

    def test_single_bad_duration_row_raises(self):
        df = _make_df()
        df.at[2, "duration"] = 0.0
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "duration"


class TestStationIdValidation:
    def test_null_station_id_raises(self):
        df = _make_df()
        df.at[0, "station_id"] = None
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "station_id"

    def test_all_null_station_ids_raises(self):
        df = _make_df()
        df["station_id"] = None
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert exc_info.value.field == "station_id"

    def test_error_message_mentions_field(self):
        df = _make_df()
        df.at[1, "station_id"] = None
        with pytest.raises(ValidationError) as exc_info:
            validate_ev_sessions(df)
        assert "station_id" in str(exc_info.value)
