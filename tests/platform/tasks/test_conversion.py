from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from ecoscope.platform.tasks.filter import TimezoneInfo
from ecoscope.platform.tasks.transformation import (
    convert_column_values_to_numeric,
    convert_column_values_to_string,
    convert_values_to_timezone,
)


def test_convert_column_values_to_string():
    df = pd.DataFrame(
        data={"value1": [1, 2, None, 4, pd.NA], "value2": ["6", "7", "8", None, "10"]},
        index=["A", "B", "C", "D", "E"],
    )
    df = convert_column_values_to_string(df, ["value1", "value2"])

    assert pd.api.types.is_string_dtype(df["value1"])
    assert pd.api.types.is_string_dtype(df["value2"])
    assert df["value1"].to_list() == ["1", "2", "None", "4", "<NA>"]
    assert df["value2"].to_list() == ["6", "7", "8", "None", "10"]


@pytest.mark.parametrize(
    "timezone, expected_dtype",
    [
        ("+03:00", "UTC+03:00"),
        ("Africa/Nairobi", "Africa/Nairobi"),
        (timezone(timedelta(hours=3)), "UTC+03:00"),
        (
            TimezoneInfo(
                label="Nairobi",
                tzCode="Africa/Nairobi",
                name="EAT",
                utc_offset="+03:00",
            ),
            "UTC+03:00",
        ),
    ],
)
def test_convert_values_to_timezone(timezone, expected_dtype):
    input_df = pd.DataFrame(
        data={
            "time": [
                pd.to_datetime("2023-06-01 15:33:00", utc=True),
                pd.to_datetime("2023-06-01 15:34:00", utc=True),
            ]
        },
        index=["A", "B"],
    )

    expected_df = pd.DataFrame(
        data={
            "time": [
                pd.to_datetime(
                    "2023-06-01 18:33:00 +0300",
                ),
                pd.to_datetime(
                    "2023-06-01 18:34:00 +0300",
                ),
            ]
        },
        index=["A", "B"],
    )

    actual_df = convert_values_to_timezone(
        df=input_df,
        timezone=timezone,
        columns=["time"],
    )

    # check the columns dtype separately as it differs
    # depending on the timezone value used to convert
    pd.testing.assert_frame_equal(expected_df, actual_df, check_dtype=False)
    assert actual_df.time.dtype == f"datetime64[ns, {expected_dtype}]"


def test_convert_column_values_to_numeric():
    df = pd.DataFrame(
        data={
            "value1": [1, 2, None, 4, pd.NA],
            "value2": [
                "6",
                {"really": ["bad", "data"]},
                "8",
                "definitely not a number",
                "10",
            ],
        },
        index=["A", "B", "C", "D", "E"],
    )
    df = convert_column_values_to_numeric(df, ["value1", "value2"])

    assert pd.api.types.is_numeric_dtype(df["value1"])
    assert pd.api.types.is_numeric_dtype(df["value2"])
    assert np.array_equal(df["value1"].to_list(), [1.0, 2.0, np.nan, 4.0, np.nan], equal_nan=True)
    assert np.array_equal(df["value2"].to_list(), [6.0, np.nan, 8.0, np.nan, 10.0], equal_nan=True)
