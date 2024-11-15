import pytest

import numpy as np
import pandas as pd

from ecoscope.io.earthranger_utils import clean_time_cols


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
