import pandas as pd
import pytest
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
