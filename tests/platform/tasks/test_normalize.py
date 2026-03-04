import pandas as pd
import pytest
from ecoscope.platform.tasks.transformation import (
    normalize_json_column,
    normalize_numeric_column,
)


def test_normalize_json_column():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["John", "Alice", "Bob"],
            "metadata": [
                {
                    "age": 30,
                    "location": {"city": "New York", "state": "NY"},
                    "profession": "journalist",
                },
                {"age": 25, "location": {"city": "San Francisco", "state": "CA"}},
                {"age": 35, "location": {"city": "Chicago", "state": "IL"}},
            ],
        }
    )

    result_df = normalize_json_column(df, "metadata")

    assert "metadata__age" in result_df
    assert "metadata__location__city" in result_df
    assert "metadata__location__state" in result_df
    assert "metadata__profession" in result_df


def test_normalize_json_column_skip_if_not_exists():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["John", "Alice", "Bob"],
        }
    )

    result_df = normalize_json_column(
        df, "non_existent_column", skip_if_not_exists=True
    )

    # The DataFrame should remain unchanged
    assert result_df.equals(df)


def test_normalize_json_column_skip_if_not_exists_false():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["John", "Alice", "Bob"],
        }
    )

    with pytest.raises(KeyError):
        normalize_json_column(df, "non_existent_column", skip_if_not_exists=False)


@pytest.mark.parametrize("output_column_name", ["normalized", None])
def test_normalize_numeric_column(output_column_name):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["John", "Alice", "Bob", "Jane"],
            "age": [5, 6, 7, 53],
        }
    )

    expected_values = [
        -0.5422260072645277,
        -0.49969847728299616,
        -0.45717094730146457,
        1.4990954318489884,
    ]

    result_df = normalize_numeric_column(
        df, column="age", output_column_name=output_column_name
    )
    if not output_column_name:
        assert result_df["age"].to_list() == expected_values
    else:
        assert result_df[output_column_name].to_list() == expected_values
        assert result_df["age"].to_list() == [5, 6, 7, 53]
