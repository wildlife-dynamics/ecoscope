import numpy as np
import pandas as pd
import pytest

from ecoscope.io.earthranger_utils import clean_time_cols, normalize_column


@pytest.fixture
def df_with_times():
    return pd.DataFrame(
        data={
            "time": [
                "2023-01-30 11:26:13.805829-08:00",
                "2023-09-27T06:16:46.158966",
                "2023-09-27T06:16:46.23-07:00",
                "2023-09-27T06:16:46.1589-07:00",
                "2023-09-27T22:00:01.23-11:00",
                "2023-09-27T06:16:46.00-07:00",
                "2023-09-27T22:00:00.00-02:00",
                pd.NA,
            ]
        },
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )


def test_clean_time_cols(df_with_times):
    with pytest.raises(AttributeError):
        df_with_times["time"].dt

    cleaned = clean_time_cols(df_with_times)
    assert pd.api.types.is_datetime64_ns_dtype(cleaned["time"])
    # Check we have our dt accessor
    df_with_times["time"].dt

    expected_times = pd.arrays.DatetimeArray._from_sequence(
        [
            "2023-01-30 19:26:13.805829+00:00",
            "2023-09-27 06:16:46.158966+00:00",
            "2023-09-27 13:16:46.230000+00:00",
            "2023-09-27 13:16:46.158900+00:00",
            "2023-09-28 09:00:01.230000+00:00",
            "2023-09-27 13:16:46+00:00",
            "2023-09-28 00:00:00+00:00",
        ]
    )

    # Since the parser resolves nan's to pd.NaT,
    # and pd.NaT != pd.NaT
    # check the nan separately from the array equality
    assert np.array_equal(expected_times, cleaned["time"].array[:-1])
    assert pd.isnull(cleaned["time"]["H"])


def test_normalize_column():
    df_with_nested_column = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "details": [
                {"zebra": "z1", "apple": "a1", "middle": "m1"},
                {"zebra": "z2", "apple": "a2", "middle": "m2"},
                {"zebra": "z3", "apple": "a3", "middle": "m3"},
            ],
            "score": [10, 20, 30],
        }
    )
    # Store original column order (excluding the one to be normalized)
    original_cols = ["id", "name", "score"]

    # Normalize the 'details' column
    normalize_column(df_with_nested_column, "details")

    # Expected new columns in alphabetical order
    expected_new_cols = ["details__apple", "details__middle", "details__zebra"]

    # Verify all expected columns exist
    assert all(col in df_with_nested_column.columns for col in original_cols)
    assert all(col in df_with_nested_column.columns for col in expected_new_cols)

    # Verify column order: original columns first, then alphabetically sorted new columns
    expected_column_order = original_cols + expected_new_cols
    assert list(df_with_nested_column.columns) == expected_column_order

    # Verify values were correctly normalized
    assert list(df_with_nested_column["details__apple"]) == ["a1", "a2", "a3"]
    assert list(df_with_nested_column["details__middle"]) == ["m1", "m2", "m3"]
    assert list(df_with_nested_column["details__zebra"]) == ["z1", "z2", "z3"]

    # Verify original columns retained their values
    assert list(df_with_nested_column["id"]) == [1, 2, 3]
    assert list(df_with_nested_column["name"]) == ["Alice", "Bob", "Charlie"]
    assert list(df_with_nested_column["score"]) == [10, 20, 30]
