import numpy as np
import pandas as pd
import pytest

from ecoscope.platform.tasks.transformation import (
    assign_value,
    fill_na,
    format_numeric_columns,
    map_values,
    map_values_with_unit,
)
from ecoscope.platform.tasks.transformation._unit import Unit


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


def test_map_values_to_an_output_column_keeps_the_source():
    df = pd.DataFrame({"season": ["Dry", "Wet", "Unknown"]})
    result = map_values(
        df,
        "season",
        {"Dry": "#f57c00", "Wet": "#255084"},
        missing_values="replace",
        replacement="#cccccc",
        output_column_name="season_colour",
    )
    assert result["season"].tolist() == ["Dry", "Wet", "Unknown"]
    assert result["season_colour"].tolist() == ["#f57c00", "#255084", "#cccccc"]


def test_map_values_preserve_into_an_output_column_copies_unmapped_values():
    df = pd.DataFrame({"status": ["open", "closed"]})
    result = map_values(df, "status", {"open": "Open"}, missing_values="preserve", output_column_name="label")
    assert result["label"].tolist() == ["Open", "closed"]
    assert result["status"].tolist() == ["open", "closed"]


@pytest.mark.parametrize("column", [[1, 2, 3], [1.0, 2.0, 3.0], np.array([1, 2, 3], dtype="int32")])
def test_map_values_numeric_column_from_text_keys(column):
    df = pd.DataFrame({"code": column})
    result = map_values(df, "code", {"1": "one", "2": "two"}, missing_values="preserve")
    assert result["code"].tolist() == ["one", "two", 3]


def test_map_values_boolean_column_from_text_keys():
    df = pd.DataFrame({"flag": [True, False]})
    result = map_values(df, "flag", {"True": "yes", "False": "no"}, missing_values="remove")
    assert result["flag"].tolist() == ["yes", "no"]


def test_map_values_warns_when_nothing_matched(caplog):
    df = pd.DataFrame({"category": ["x", "y"]})
    with caplog.at_level("WARNING"):
        map_values(df, "category", {"A": "Alpha"}, missing_values="preserve")
    assert "none of the values in 'category' matched" in caplog.text


def test_map_values_no_warning_when_something_matched(caplog, sample_df, value_map):
    with caplog.at_level("WARNING"):
        map_values(sample_df, "category", value_map, missing_values="preserve")
    assert "matched" not in caplog.text


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


def test_map_values_with_unit_converts_and_formats():
    df = pd.DataFrame({"dist": [1500.0, 2750.5, 0.0]})
    result = map_values_with_unit(
        df,
        input_column_name="dist",
        output_column_name="dist_km",
        original_unit=Unit.METER,
        new_unit=Unit.KILOMETER,
        decimal_places=2,
    )
    assert result["dist_km"].tolist() == ["1.50 km", "2.75 km", "0.00 km"]


def test_map_values_with_unit_same_unit_skips_conversion():
    df = pd.DataFrame({"dist": [1.234, 5.678]})
    result = map_values_with_unit(
        df,
        input_column_name="dist",
        output_column_name="dist_fmt",
        original_unit=Unit.METER,
        new_unit=Unit.METER,
        decimal_places=1,
    )
    # Same-unit short-circuit: no scaling, just decimal formatting with the unit suffix.
    assert result["dist_fmt"].tolist() == ["1.2 m", "5.7 m"]


def test_map_values_with_unit_no_unit_at_all():
    df = pd.DataFrame({"v": [3.14, 2.71]})
    result = map_values_with_unit(
        df,
        input_column_name="v",
        output_column_name="v_fmt",
        decimal_places=1,
    )
    # No unit specified: format only, no suffix.
    assert result["v_fmt"].tolist() == ["3.1", "2.7"]


def test_map_values_with_unit_speed_conversion():
    df = pd.DataFrame({"speed": [10.0, 0.0]})
    result = map_values_with_unit(
        df,
        input_column_name="speed",
        output_column_name="speed_kmh",
        original_unit=Unit.METERS_PER_SECOND,
        new_unit=Unit.KILOMETERS_PER_HOUR,
        decimal_places=1,
    )
    assert result["speed_kmh"].tolist() == ["36.0 km/h", "0.0 km/h"]


def test_map_values_with_unit_area_conversion():
    df = pd.DataFrame({"area": [1_000_000.0, 500_000.0, 0.0]})
    result = map_values_with_unit(
        df,
        input_column_name="area",
        output_column_name="area_km2",
        original_unit=Unit.SQUARE_METER,
        new_unit=Unit.SQUARE_KILOMETER,
        decimal_places=2,
    )
    assert result["area_km2"].tolist() == ["1.00 km²", "0.50 km²", "0.00 km²"]


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


def test_format_numeric_columns_fixed_decimals_in_place():
    df = pd.DataFrame({"rate": [0.5, 1.23456, np.nan], "effort": [3.5, 10.0, 0.004], "label": ["a", "b", "c"]})
    result = format_numeric_columns(df, columns=["rate", "effort", "label"], decimal_places=2)
    assert result["rate"].tolist() == ["0.50", "1.23", ""]
    assert result["effort"].tolist() == ["3.50", "10.00", "0.00"]
    assert result["label"].tolist() == ["a", "b", "c"]
    assert result.index.tolist() == [0, 1, 2]
    # the input frame is untouched
    assert df["rate"].tolist()[:2] == [0.5, 1.23456]


def test_format_numeric_columns_skips_absent_unless_asked():
    df = pd.DataFrame({"count": [3.0, 4.0]})
    result = format_numeric_columns(df, columns=["count", "event_sum"], decimal_places=0)
    assert result["count"].tolist() == ["3", "4"]
    assert "event_sum" not in result.columns
    with pytest.raises(KeyError):
        format_numeric_columns(df, columns=["event_sum"], raise_if_not_found=True)


def test_format_numeric_columns_empty_frame():
    df = pd.DataFrame({"rate": pd.Series([], dtype=float)})
    result = format_numeric_columns(df, columns=["rate"])
    assert result.empty and list(result.columns) == ["rate"]
