import pandas as pd
from ecoscope.platform.tasks.transformation import (
    extract_column_as_type,
    extract_value_from_json_column,
)
from ecoscope.platform.tasks.transformation._extract import (
    FieldType,
    extract_value_as_type,
)


def test_extract_value_as_str():
    assert extract_value_as_type("value1", FieldType.STRING) == "value1"
    assert extract_value_as_type(1, FieldType.STRING) == "1"
    assert extract_value_as_type(None, FieldType.STRING) is None


def test_extract_value_as_float():
    assert extract_value_as_type("0.01", FieldType.FLOAT) == 0.01
    assert extract_value_as_type(1, FieldType.FLOAT) == 1
    assert extract_value_as_type("invalid", FieldType.FLOAT) is None


def test_extract_value_as_bool():
    assert extract_value_as_type("TRUE", FieldType.BOOL)
    assert extract_value_as_type(True, FieldType.BOOL)
    assert extract_value_as_type(1, FieldType.BOOL)
    assert not extract_value_as_type(0, FieldType.BOOL)
    assert extract_value_as_type("invalid", FieldType.BOOL) is None


def test_extract_value_as_datetime():
    assert extract_value_as_type("2023-01-01", FieldType.DATETIME) == pd.to_datetime("2023-01-01", utc=True)
    assert extract_value_as_type("invalid", FieldType.DATETIME) is None


def test_extract_value_as_date():
    assert extract_value_as_type("2023-01-01", FieldType.DATE) == pd.to_datetime("2023-01-01", utc=True).date()
    assert extract_value_as_type("invalid", FieldType.DATE) is None


def test_extract_value_as_json():
    assert extract_value_as_type({"nested": {"value": 42}}, FieldType.JSON) == '{"nested": {"value": 42}}'
    assert extract_value_as_type('{"nested": {"value": 42}}', FieldType.JSON) == '{"nested": {"value": 42}}'
    assert extract_value_as_type([1, 2], FieldType.JSON) == "[1, 2]"


def test_extract_value_as_series():
    s = extract_value_as_type({"start": "2023-01-01", "end": "2023-12-31"}, FieldType.SERIES)
    pd.testing.assert_series_equal(s, pd.Series(["2023-01-01", "2023-12-31"], index=["start", "end"]))

    s = extract_value_as_type(["start", "end"], FieldType.SERIES)
    pd.testing.assert_series_equal(s, pd.Series(["start", "end"]))


def test_extract_column_as_type():
    data = {
        "str": ["value1", "value4", "value5"],
        "series": [
            {"start": "2023-01-01", "end": "2023-12-31"},
            {"start": "2023-01-01", "end": "2023-12-31"},
            {"start": "2023-01-01", "end": "2023-12-31"},
        ],
    }
    df = pd.DataFrame(data)

    # Test case 1: Extract string values
    result_df = extract_column_as_type(
        df,
        column_name="str",
        output_type=FieldType.STRING,
        output_column_name="extracted_value",
    )
    expected_values = ["value1", "value4", "value5"]
    assert result_df["extracted_value"].tolist() == expected_values

    # Test case 2: Extract series
    result_df = extract_column_as_type(
        df,
        column_name="series",
        output_type=FieldType.SERIES,
        output_column_name="extracted_value.",
    )
    expected_start = ["2023-01-01", "2023-01-01", "2023-01-01"]
    expected_end = ["2023-12-31", "2023-12-31", "2023-12-31"]
    assert result_df["extracted_value.start"].tolist() == expected_start
    assert result_df["extracted_value.end"].tolist() == expected_end


def test_extract_value_from_json_column():
    data = {
        "json_column": [
            {
                "field1": "value1",
                "field2": "value2",
                "field3": {"start": "2023-01-01", "end": "2023-12-31"},
            },
            {
                "field2": "value4",
                "field3": {"start": "2023-01-01", "end": "2023-12-31"},
            },
            {
                "field1": "value5",
                "field3": {"start": "2023-01-01", "end": "2023-12-31"},
            },
        ]
    }
    df = pd.DataFrame(data)

    # Test case 1: Extract 'field1' or 'field2' values
    result_df = extract_value_from_json_column(
        df,
        column_name="json_column",
        field_name_options=["field1", "field2"],
        output_type=FieldType.STRING,
        output_column_name="extracted_value",
    )
    expected_values = ["value1", "value4", "value5"]
    assert result_df["extracted_value"].tolist() == expected_values

    # Test case 2: Missing field
    result_df = extract_value_from_json_column(
        df,
        column_name="json_column",
        field_name_options=["field2"],
        output_type=FieldType.STRING,
        output_column_name="extracted_value",
    )
    expected_values = ["value2", "value4", None]
    assert result_df["extracted_value"].tolist() == expected_values

    # Test case 3: Series
    result_df = extract_value_from_json_column(
        df,
        column_name="json_column",
        field_name_options=["field3"],
        output_type=FieldType.SERIES,
        output_column_name="extracted_value.",
    )
    assert result_df["extracted_value.start"][0] == "2023-01-01"
    assert result_df["extracted_value.end"][0] == "2023-12-31"
