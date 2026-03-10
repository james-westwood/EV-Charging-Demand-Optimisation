"""Tests for src/data/validators/report.py."""
import pandas as pd
import pytest

from src.data.validators.report import validate_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_carbon_intensity_df(
    n: int = 3,
    intensity_actual: float = 200.0,
) -> pd.DataFrame:
    """Build a carbon-intensity DataFrame with n rows at 30-min intervals."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    return pd.DataFrame(
        {
            "settlement_period": [
                base + pd.Timedelta(minutes=30 * i) for i in range(n)
            ],
            "intensity_actual": [intensity_actual] * n,
            "intensity_forecast": [180.0] * n,
        }
    )


def _make_ev_sessions_df(
    n: int = 3,
    energy_kwh: float = 20.0,
    duration: float = 60.0,
) -> pd.DataFrame:
    """Build an ev-sessions DataFrame with n rows."""
    base = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    rows = []
    for i in range(n):
        arrival = base + pd.Timedelta(hours=i * 2)
        departure = arrival + pd.Timedelta(minutes=90)
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


# ---------------------------------------------------------------------------
# Core acceptance-criteria test
# ---------------------------------------------------------------------------

class TestValidateAllReport:
    def test_three_bad_rows_across_two_sources_gives_two_keys(self):
        """3 bad rows across 2 sources → report has 2 keys with correct error counts."""
        # Source 1: carbon_intensity with 2 rows having intensity > 800
        ci_df = _make_carbon_intensity_df(n=4, intensity_actual=200.0)
        ci_df.at[1, "intensity_actual"] = 999.0  # bad row 1
        ci_df.at[3, "intensity_actual"] = 850.0  # bad row 2

        # Source 2: ev_sessions with 1 row having energy_kwh <= 0
        ev_df = _make_ev_sessions_df(n=3, energy_kwh=20.0)
        ev_df.at[2, "energy_kwh"] = -5.0  # bad row 3

        report = validate_all({"carbon_intensity": ci_df, "ev_sessions": ev_df})

        assert len(report) == 2, f"Expected 2 keys, got {list(report.keys())}"
        assert "carbon_intensity" in report
        assert "ev_sessions" in report
        assert len(report["carbon_intensity"]) == 2, (
            f"Expected 2 errors for carbon_intensity, got {len(report['carbon_intensity'])}"
        )
        assert len(report["ev_sessions"]) == 1, (
            f"Expected 1 error for ev_sessions, got {len(report['ev_sessions'])}"
        )

    def test_all_valid_sources_returns_empty_dict(self):
        report = validate_all(
            {
                "carbon_intensity": _make_carbon_intensity_df(n=3),
                "ev_sessions": _make_ev_sessions_df(n=3),
            }
        )
        assert report == {}

    def test_only_failing_sources_appear_in_report(self):
        good_ci = _make_carbon_intensity_df(n=3)
        bad_ev = _make_ev_sessions_df(n=3)
        bad_ev.at[0, "energy_kwh"] = 0.0

        report = validate_all({"carbon_intensity": good_ci, "ev_sessions": bad_ev})

        assert "carbon_intensity" not in report
        assert "ev_sessions" in report

    def test_unknown_source_is_skipped_silently(self):
        report = validate_all({"unknown_source": pd.DataFrame({"x": [1, 2]})})
        assert report == {}

    def test_empty_dataframes_produce_no_errors(self):
        report = validate_all(
            {
                "carbon_intensity": pd.DataFrame(
                    columns=["settlement_period", "intensity_actual", "intensity_forecast"]
                ),
                "ev_sessions": pd.DataFrame(
                    columns=["station_id", "arrival_time", "departure_time", "energy_kwh", "duration"]
                ),
            }
        )
        assert report == {}


# ---------------------------------------------------------------------------
# Error message content
# ---------------------------------------------------------------------------

class TestErrorMessageContent:
    def test_error_messages_are_strings(self):
        bad_ev = _make_ev_sessions_df(n=2)
        bad_ev.at[0, "energy_kwh"] = -1.0

        report = validate_all({"ev_sessions": bad_ev})

        assert all(isinstance(msg, str) for msg in report["ev_sessions"])

    def test_error_messages_mention_field(self):
        bad_ci = _make_carbon_intensity_df(n=2)
        bad_ci.at[0, "intensity_actual"] = 999.0

        report = validate_all({"carbon_intensity": bad_ci})

        assert any("intensity_actual" in msg for msg in report["carbon_intensity"])

    def test_multiple_errors_are_distinct(self):
        bad_ci = _make_carbon_intensity_df(n=3)
        bad_ci.at[0, "intensity_actual"] = 999.0
        bad_ci.at[2, "intensity_actual"] = 850.0

        report = validate_all({"carbon_intensity": bad_ci})

        errors = report["carbon_intensity"]
        assert len(errors) == 2
        assert errors[0] != errors[1]


# ---------------------------------------------------------------------------
# Source-specific validation
# ---------------------------------------------------------------------------

class TestCarbonIntensityErrors:
    def test_out_of_range_intensity_reported(self):
        df = _make_carbon_intensity_df(n=3)
        df.at[1, "intensity_actual"] = 999.0

        report = validate_all({"carbon_intensity": df})

        assert "carbon_intensity" in report
        assert len(report["carbon_intensity"]) == 1

    def test_null_settlement_period_reported(self):
        df = _make_carbon_intensity_df(n=3)
        df.at[0, "settlement_period"] = None

        report = validate_all({"carbon_intensity": df})

        assert "carbon_intensity" in report


class TestEvSessionsErrors:
    def test_negative_energy_reported(self):
        df = _make_ev_sessions_df(n=3)
        df.at[0, "energy_kwh"] = -10.0

        report = validate_all({"ev_sessions": df})

        assert "ev_sessions" in report

    def test_departure_before_arrival_reported(self):
        df = _make_ev_sessions_df(n=3)
        df.at[1, "departure_time"] = df.at[1, "arrival_time"] - pd.Timedelta(minutes=10)

        report = validate_all({"ev_sessions": df})

        assert "ev_sessions" in report

    def test_null_station_id_reported(self):
        df = _make_ev_sessions_df(n=2)
        df.at[0, "station_id"] = None

        report = validate_all({"ev_sessions": df})

        assert "ev_sessions" in report
