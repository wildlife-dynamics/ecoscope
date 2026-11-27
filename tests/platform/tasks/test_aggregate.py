import pandas as pd
import pytest

from ecoscope.platform.tasks.analysis import (
    apply_arithmetic_operation,
    dataframe_column_first_unique,
    dataframe_column_max,
    dataframe_column_mean,
    dataframe_column_min,
    dataframe_column_nunique,
    dataframe_column_percentile,
    dataframe_column_sum,
    dataframe_count,
)


def test_count():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_count(df)
    assert result == 3


def test_mean():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_mean(df, "data")
    assert result == 2.0


def test_sum():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_sum(df, "data")
    assert result == 6


def test_max():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_max(df, "data")
    assert result == 3


def test_min():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_min(df, "data")
    assert result == 1


def test_nunique():
    df = pd.DataFrame({"data": [1, 2, 3, 1]})
    result = dataframe_column_nunique(df, "data")
    assert result == 3


def test_first_unique():
    df = pd.DataFrame({"data": [1, 2, 3, 1]})
    result = dataframe_column_first_unique(df, "data")
    assert result == 1


def test_first_unique_is_a_plain_python_value():
    result = dataframe_column_first_unique(pd.DataFrame({"year": [2024, 2023]}), "year")
    assert result == 2024 and type(result) is int


def test_first_unique_text_column():
    assert dataframe_column_first_unique(pd.DataFrame({"name": ["Tsavo", "Mara"]}), "name") == "Tsavo"


def test_first_unique_skips_empty_cells():
    df = pd.DataFrame({"name": [None, float("nan"), "Tsavo"]})
    assert dataframe_column_first_unique(df, "name") == "Tsavo"


def test_first_unique_as_text():
    df = pd.DataFrame({"year": [2024.0, 2023.0]})
    assert dataframe_column_first_unique(df, "year", as_text=True) == "2024.0"
    assert dataframe_column_first_unique(pd.DataFrame({"n": [7]}), "n", as_text=True) == "7"


def test_first_unique_timestamp():
    df = pd.DataFrame({"t": pd.to_datetime(["2025-01-02", "2025-01-01"])})
    assert dataframe_column_first_unique(df, "t") == pd.Timestamp("2025-01-02")


@pytest.mark.parametrize(
    "df, column, message",
    [
        (pd.DataFrame(), "data", "the table is empty"),
        (pd.DataFrame({"data": [1]}), "missing", "'missing' not found"),
        (pd.DataFrame({"data": [None, None]}), "data", "no non-empty value"),
    ],
)
def test_first_unique_errors(df, column, message):
    with pytest.raises(ValueError, match=message):
        dataframe_column_first_unique(df, column)


def test_percentile():
    df = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    result = dataframe_column_percentile(df, "data", 50)
    assert result == 5.5


@pytest.mark.parametrize(
    "a, b, operation, expected",
    [
        (1, 2, "add", 3),
        (3, 2, "subtract", 1),
        (2, 3, "multiply", 6),
        (6, 3, "divide", 2),
        (7, 3, "floor_divide", 2),
        (7, 3, "modulo", 1),
        (2, 3, "power", 8),
        (2, 3, "min", 2),
        (2, 3, "max", 3),
    ],
)
def test_apply_arithmetic_operation(a, b, operation, expected):
    result = apply_arithmetic_operation(a, b, operation)
    assert result == expected
