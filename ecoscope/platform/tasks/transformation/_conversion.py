import datetime
from typing import Annotated, cast

import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.filter import TimezoneInfo


@register()
def convert_column_values_to_string(
    df: AnyDataFrame,
    columns: Annotated[list[str], Field(description="The columns to convert.")],
) -> AnyDataFrame:
    """
    Casts the values of the listed columns to type string
    None and NaN values will also be converted to string

    Args:
        df (AnyDataFrame): The input DataFrame.
        columns (list[str]): List of columns to cast to string.

    Returns:
        AnyDataFrame: The modified DataFrame.
    """
    for col in columns:
        df[col] = df[col].astype(str)

    return cast(AnyDataFrame, df)


@register()
def convert_values_to_timezone(
    df: AnyDataFrame,
    timezone: Annotated[str | datetime.tzinfo | TimezoneInfo, Field()],
    columns: Annotated[list[str], Field(description="The columns to convert.")],
) -> AnyDataFrame:
    """
    Converts the listed columns in the df to the timezone provided
    NOTE: Timezone naive timestamps are ignored
    Args:
        df (AnyDataFrame): The input DataFrame.
        timezone (str | datetime.tzinfo | TimezoneInfo): The timezone to convert to
        columns (list[str]): List of columns to cast to string.

    Returns:
        AnyDataFrame: The modified DataFrame.
    """
    if isinstance(timezone, TimezoneInfo):
        timezone = timezone.utc_offset
    for col in columns:
        if col in df and isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_convert(timezone).dt.as_unit("ns")

    return cast(AnyDataFrame, df)


@register()
def convert_column_values_to_numeric(
    df: AnyDataFrame,
    columns: Annotated[list[str], Field(description="The columns to convert.")],
) -> AnyDataFrame:
    """
    Casts the values of the listed columns to numbers
    Values that cannot be casted will be converted to NaN

    Args:
        df (AnyDataFrame): The input DataFrame.
        columns (list[str]): List of columns to cast.

    Returns:
        AnyDataFrame: The modified DataFrame.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return cast(AnyDataFrame, df)
