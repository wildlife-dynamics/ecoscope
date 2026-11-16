import pandas as pd
import pytest

from ecoscope.platform.tasks.transformation import merge_two_dataframes


@pytest.fixture
def left():
    return pd.DataFrame({"id": [1, 2, 3], "site": ["a", "b", "c"], "value": [10, 20, 30]})


@pytest.fixture
def right():
    return pd.DataFrame({"id": [2, 3, 4], "user_id": [2, 3, 4], "value": [200, 300, 400]})


def test_inner_merge_on_a_shared_column_suffixes_overlaps(left, right):
    out = merge_two_dataframes(left, right[["id", "value"]], on="id")
    assert out["id"].tolist() == [2, 3]
    assert out["value_x"].tolist() == [20, 30]
    assert out["value_y"].tolist() == [200, 300]


def test_merge_on_a_list_of_columns(left):
    right = pd.DataFrame({"id": [1, 1], "site": ["a", "z"], "n": [5, 6]})
    out = merge_two_dataframes(left, right, how="inner", on=["id", "site"])
    assert out[["id", "site", "n"]].values.tolist() == [[1, "a", 5]]


def test_left_merge_with_differently_named_keys_drops_the_right_key(left, right):
    out = merge_two_dataframes(left, right[["user_id", "value"]], how="left", left_on="id", right_on="user_id")
    assert "user_id" not in out
    assert out["id"].tolist() == [1, 2, 3]
    assert out["value_y"].isna().tolist() == [True, False, False]


def test_right_merge_with_differently_named_keys_drops_the_left_key(left, right):
    out = merge_two_dataframes(
        left.rename(columns={"id": "left_id"}), right, how="right", left_on="left_id", right_on="user_id"
    )
    assert "left_id" not in out
    assert out["user_id"].tolist() == [2, 3, 4]


def test_inner_merge_keeps_both_differently_named_keys(left, right):
    out = merge_two_dataframes(left, right[["user_id"]], how="inner", left_on="id", right_on="user_id")
    assert out[["id", "user_id"]].values.tolist() == [[2, 2], [3, 3]]


def test_outer_merge_fills_nulls(left, right):
    out = merge_two_dataframes(left[["id", "site"]], right[["id", "value"]], how="outer", on="id", fillna_value=0)
    assert out["id"].tolist() == [1, 2, 3, 4]
    assert out["site"].tolist() == ["a", "b", "c", 0]
    assert out["value"].tolist() == [0, 200, 300, 400]


def test_merge_on_indexes(left, right):
    out = merge_two_dataframes(left[["site"]], right[["user_id"]], left_index=True, right_index=True)
    assert out.index.tolist() == [0, 1, 2]
    assert out.columns.tolist() == ["site", "user_id"]


def test_cross_merge(left, right):
    out = merge_two_dataframes(left[["site"]], right[["user_id"]], how="cross")
    assert len(out) == 9
