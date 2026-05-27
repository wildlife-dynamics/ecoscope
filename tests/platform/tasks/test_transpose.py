import pandas as pd

from ecoscope.platform.tasks.transformation import transpose


def test_transpose_default_range_index() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    result = transpose(df)

    assert list(result.columns) == ["0", "1"]
    assert list(result.index) == ["a", "b"]


def test_transpose_named_index_with_transposed_column_name() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["x", "y"])

    result = transpose(df, transposed_column_name="orig")

    assert "orig" in result.columns
    assert set(result["orig"].tolist()) == {"a", "b"}


def test_transpose_range_index_with_transposed_column_name() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    result = transpose(df, transposed_column_name="orig")

    assert "orig" in result.columns
    assert "0" in result.columns and "1" in result.columns
