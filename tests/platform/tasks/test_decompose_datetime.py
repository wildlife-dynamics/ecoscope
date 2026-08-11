import pandas as pd

from ecoscope.platform.tasks.transformation import decompose_datetime


def test_extracts_components_with_default_prefix():
    df = pd.DataFrame({"segment_start": ["2026-01-15 08:30:00", "2026-02-20 17:00:00"]})
    result = decompose_datetime(df, datetime_column="segment_start", components=["date", "year"])
    assert list(result["segment_start_date"]) == [pd.Timestamp("2026-01-15").date(), pd.Timestamp("2026-02-20").date()]
    assert list(result["segment_start_year"]) == [2026, 2026]
    assert "segment_start" in result.columns


def test_remove_source_and_custom_prefix():
    df = pd.DataFrame({"t": ["2026-01-15"]})
    result = decompose_datetime(df, datetime_column="t", components=["month"], remove_source=True, column_prefix="ts")
    assert "t" not in result.columns
    assert list(result["ts_month"]) == [1]


def test_week_component_uses_isocalendar():
    df = pd.DataFrame({"t": ["2026-01-15", "2026-12-28"]})
    result = decompose_datetime(df, datetime_column="t", components=["week"])
    assert list(result["t_week"]) == [3, 53]
