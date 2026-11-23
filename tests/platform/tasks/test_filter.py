import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.platform.tasks.transformation._filter import (
    ComparisonOperator,
    filter_df,
)


@pytest.fixture
def df():
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})


def test_filter_equal(df):
    result = filter_df(df, "A", ComparisonOperator.EQUAL, 3)
    expected = df[df["A"] == 3]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_ge(df):
    result = filter_df(df, "A", ComparisonOperator.GE, 3)
    expected = df[df["A"] >= 3]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_gt(df):
    result = filter_df(df, "A", ComparisonOperator.GT, 3)
    expected = df[df["A"] > 3]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_le(df):
    result = filter_df(df, "A", ComparisonOperator.LE, 3)
    expected = df[df["A"] <= 3]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_ne(df):
    result = filter_df(df, "A", ComparisonOperator.NE, 3)
    expected = df[df["A"] != 3]
    pd.testing.assert_frame_equal(result, expected)


@pytest.fixture
def events():
    return gpd.GeoDataFrame(
        {
            "event_type": ["carcass", "snare", "carcass", "", None],
            "time": pd.to_datetime(
                ["2025-01-06 08:00", "2025-01-08 09:30", "2025-01-13 10:00", "2025-01-14 11:00", "2025-01-15 12:00"]
            ).tz_localize("UTC"),
            "dist_meters": [100.0, 250.5, 30.0, np.nan, 5.0],
            "is_collected": [True, False, True, False, True],
        },
        geometry=[Point(36.8, -1.3), Point(36.9, -1.4), Point(37.0, -1.5), Point(37.1, -1.6), None],
        crs="EPSG:4326",
    )


@pytest.mark.parametrize(
    "column, op, value, values, kept",
    [
        ("event_type", "is_any_of", "", ["snare", "carcass"], [0, 1, 2]),
        ("event_type", "is_none_of", "", ["carcass"], [1, 3, 4]),
        ("event_type", "is_any_of", "", None, []),
        ("event_type", "is_none_of", "", None, [0, 1, 2, 3, 4]),
        ("event_type", "is_empty", "", None, [3, 4]),
        ("event_type", "is_not_empty", "", None, [0, 1, 2]),
        ("dist_meters", "is_empty", "", None, [3]),
        ("geometry", "is_not_empty", "", None, [0, 1, 2, 3]),
        ("event_type", "equal", "snare", None, [1]),
        ("event_type", "ne", "carcass", None, [1, 3, 4]),
        ("dist_meters", "gt", "30", None, [0, 1]),
        ("dist_meters", "ge", "30.0", None, [0, 1, 2]),
        ("dist_meters", "ne", "30", None, [0, 1, 3, 4]),
        ("dist_meters", "is_any_of", "", ["5", "100"], [0, 4]),
        ("time", "ge", "2025-01-13", None, [2, 3, 4]),
        ("time", "equal", "2025-01-08T12:30:00+03:00", None, [1]),
        ("is_collected", "equal", "true", None, [0, 2, 4]),
        ("is_collected", "is_any_of", "", ["False"], [1, 3]),
    ],
)
def test_filter_reads_operands_as_the_column_type(events, column, op, value, values, kept):
    result = filter_df(events, column, ComparisonOperator(op), value, values=values)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.index.tolist() == kept


@pytest.mark.parametrize(
    "column, value, kind",
    [("dist_meters", "abc", "number"), ("time", "yesterday-ish", "date"), ("is_collected", "maybe", "boolean")],
)
def test_filter_operand_not_of_the_column_type_fails(events, column, value, kind):
    with pytest.raises(ValueError, match=rf"{value!r}.*{kind} column '{column}'"):
        filter_df(events, column, ComparisonOperator.EQUAL, value)


def test_filter_comparison_without_a_value_fails_on_a_number_column(events):
    with pytest.raises(ValueError, match="cannot compare value '' with number column 'dist_meters'"):
        filter_df(events, "dist_meters", ComparisonOperator.GT)


def test_filter_geometry_column_only_tests_emptiness(events):
    with pytest.raises(ValueError, match="geometry column 'geometry' can only be tested"):
        filter_df(events, "geometry", ComparisonOperator.EQUAL, "x")
