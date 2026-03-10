import pandas as pd
import pytest

from ecoscope.platform.tasks.transformation import (
    map_columns,
    reorder_columns,
    title_case_columns_by_prefix,
)
from ecoscope.platform.tasks.transformation._mapping import RenameColumn


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
    return pd.DataFrame(data)


def test_drop_columns(sample_dataframe):
    """Test that columns are correctly dropped."""
    result_df = map_columns(sample_dataframe, drop_columns=["A"], retain_columns=[], rename_columns={})
    assert "A" not in result_df.columns


def test_drop_columns_error(sample_dataframe):
    """Test raising error if a column does not exist."""
    with pytest.raises(KeyError):
        map_columns(
            sample_dataframe,
            drop_columns=["NOT_EXIST"],
            retain_columns=[],
            rename_columns={},
        )


def test_drop_columns_ignore_missing(sample_dataframe):
    """Test that missing columns are ignored when raise_if_not_found is False."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=["A", "NOT_EXIST"],
        retain_columns=[],
        rename_columns={},
        raise_if_not_found=False,
    )
    assert "A" not in result_df.columns
    assert list(result_df.columns) == ["B", "C"]


def test_retain_columns(sample_dataframe):
    """Test that only specified columns are retained."""
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=["B"], rename_columns={})
    assert list(result_df.columns) == ["B"]


def test_retain_columns_respects_order(sample_dataframe):
    """Test that only specified columns are retained."""
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=["B", "A"], rename_columns={})
    assert list(result_df.columns) == ["B", "A"]


def test_retain_columns_error(sample_dataframe):
    """Test raising error if a column does not exist."""
    with pytest.raises(KeyError):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=["NOT_EXIST"],
            rename_columns={},
        )


def test_rename_columns(sample_dataframe):
    """Test that columns are correctly renamed."""
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns={"B": "Z"})
    assert "Z" in result_df.columns and "B" not in result_df.columns


def test_rename_columns_error(sample_dataframe):
    """Test raising error if a column does not exist."""
    with pytest.raises(KeyError, match=r"Columns \['NOT_EXIST'\] not all found in DataFrame\."):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=[],
            rename_columns={"NOT_EXIST": "Z"},
        )


def test_rename_columns_with_list(sample_dataframe):
    """Test that columns are correctly renamed using a list of RenameColumn objects."""
    rename_list = [
        RenameColumn(original_name="B", new_name="Z"),
        RenameColumn(original_name="C", new_name="Y"),
    ]
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns=rename_list)
    assert "Z" in result_df.columns and "B" not in result_df.columns
    assert "Y" in result_df.columns and "C" not in result_df.columns
    assert "A" in result_df.columns


def test_rename_columns_with_list_single_column(sample_dataframe):
    """Test renaming a single column using a list of RenameColumn objects."""
    rename_list = [RenameColumn(original_name="A", new_name="X")]
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns=rename_list)
    assert "X" in result_df.columns and "A" not in result_df.columns
    assert list(result_df.columns) == ["X", "B", "C"]


def test_rename_columns_with_list_error(sample_dataframe):
    """Test raising error if a column does not exist when using list format."""
    rename_list = [RenameColumn(original_name="NOT_EXIST", new_name="Z")]
    with pytest.raises(KeyError, match=r"Columns \['NOT_EXIST'\] not all found in DataFrame\."):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=[],
            rename_columns=rename_list,
        )


def test_map_columns_with_rename_list(sample_dataframe):
    """Test that columns are correctly mapped when using list format for renaming."""
    rename_list = [RenameColumn(original_name="B", new_name="Z")]
    result_df = map_columns(
        sample_dataframe,
        drop_columns=["C"],
        retain_columns=["B"],
        rename_columns=rename_list,
    )
    assert list(result_df.columns) == ["Z"]


def test_map_columns(sample_dataframe):
    """Test that columns are correctly mapped."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=["C"],
        retain_columns=["B"],
        rename_columns={"B": "Z"},
    )
    assert list(result_df.columns) == ["Z"]


def test_title_case_columns_by_prefix():
    df = pd.DataFrame(
        data={
            "a_value": [1, 2, 3],
            "extra__another_value": [4, 5, 6],
            "extra__a_third_value": [7, 8, 9],
        }
    )

    df = title_case_columns_by_prefix(df, prefix="extra__")
    assert df.columns.to_list() == [
        "a_value",
        "Another Value",
        "A Third Value",
    ]


def test_reorder_columns():
    df = pd.DataFrame(
        data={
            "a_value": [1, 2, 3],
            "another_value": [4, 5, 6],
            "a_third_value": [7, 8, 9],
        }
    )

    df = reorder_columns(df, columns=["a_third_value", "another_value"])
    assert df.columns.to_list() == [
        "a_third_value",
        "another_value",
        "a_value",
    ]
