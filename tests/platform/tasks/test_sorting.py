import numpy as np
import pandas as pd
import pytest
from ecoscope.platform.tasks.transformation import sort_values


@pytest.fixture
def unsorted_df():
    return pd.DataFrame({"value": [500, 200, 300, 150, np.nan]})


@pytest.mark.parametrize(
    "ascending, na_pos",
    [
        (True, "last"),
        (True, "first"),
        (False, "last"),
        (False, "first"),
    ],
)
def test_sort_values(unsorted_df, ascending, na_pos):
    sorted_df = sort_values(unsorted_df, column_name="value", ascending=ascending, na_position=na_pos)
    expected = unsorted_df.sort_values(by="value", ascending=ascending, na_position=na_pos)
    pd.testing.assert_frame_equal(sorted_df, expected)
