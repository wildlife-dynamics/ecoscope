import numpy as np
import pandas as pd
import pytest

from ecoscope.platform.tasks.transformation import (
    assign_value,
    fill_na,
    map_values,
)


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame."""
    return pd.DataFrame({"category": ["A", "B", "C", "D", None, pd.NA]})


@pytest.fixture
def value_map():
    """Fixture to create a sample value map."""
    return {"A": "Alpha", "B": "Beta"}


def test_map_values_remove(sample_df, value_map):
    """Test value mapping without preserving values not in the map."""
    result_df = map_values(sample_df, "category", value_map, missing_values="remove")
    expected_df = pd.DataFrame(
        {
            "category": [
                "Alpha",
                "Beta",
                None,
                None,
                None,
                None,
            ]  # Assuming non-mapped values are set to None
        }
    )
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_map_values_preserve(sample_df, value_map):
    """Test value mapping with preserving values not in the map."""
    result_df = map_values(sample_df, "category", value_map, missing_values="preserve")
    expected_df = pd.DataFrame(
        {"category": ["Alpha", "Beta", "C", "D", None, pd.NA]}  # Non-mapped values are preserved
    )
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_map_values_replace(sample_df, value_map):
    """Test value mapping with replacement value."""
    result_df = map_values(sample_df, "category", value_map, missing_values="replace", replacement="None")
    expected_df = pd.DataFrame({"category": ["Alpha", "Beta", "None", "None", "None", "None"]})
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_assign_value():
    df = pd.DataFrame({"col1": [1, 2, 3]})
    result = assign_value(df, "col1", 5)
    expected = pd.DataFrame({"col1": [5, 5, 5]})
    pd.testing.assert_frame_equal(result, expected)

    result = assign_value(df, "col2", 6)
    expected = pd.DataFrame({"col1": [5, 5, 5], "col2": [6, 6, 6]})
    pd.testing.assert_frame_equal(result, expected)

    result = assign_value(df, "col2", 10, noop_if_column_exists=True)
    pd.testing.assert_frame_equal(result, expected)


def test_fill_na_numeric():
    column_with_nans = pd.Series([np.nan, 14.0, np.nan, 16.0])
    column_with_nans_filled = pd.Series([0.0, 14.0, 0.0, 16.0])

    df_with_nans = pd.DataFrame(
        {
            "col_a": column_with_nans,
            "col_b": column_with_nans,
        },
    )

    result = fill_na(df_with_nans, value=0.0)
    assert result["col_a"].equals(column_with_nans_filled)
    assert result["col_b"].equals(column_with_nans_filled)

    result = fill_na(df_with_nans, value=0.0, columns=["col_a"])
    assert result["col_a"].equals(column_with_nans_filled)
    assert result["col_b"].equals(column_with_nans)


def test_fill_na_string():
    column_with_nans = pd.Series([np.nan, "Value", np.nan, "More Value"])
    column_with_nans_filled = pd.Series(["None", "Value", "None", "More Value"])

    df_with_nans = pd.DataFrame(
        {
            "col_a": column_with_nans,
            "col_b": column_with_nans,
        },
    )

    result = fill_na(df_with_nans, value="None")
    assert result["col_a"].equals(column_with_nans_filled)
    assert result["col_b"].equals(column_with_nans_filled)

    result = fill_na(df_with_nans, value="None", columns=["col_a"])
    assert result["col_a"].equals(column_with_nans_filled)
    assert result["col_b"].equals(column_with_nans)
