import datetime
import logging
from typing import Annotated, cast

import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.filter import TimezoneInfo

logger = logging.getLogger(__name__)


@register()
def convert_column_values_to_string(
    df: AnyDataFrame,
    columns: Annotated[list[str], Field(description="The columns to convert.")],
    skip_missing: Annotated[
        bool,
        Field(description="Skip listed columns that aren't in the table instead of failing."),
    ] = False,
) -> AnyDataFrame:
    """
    Casts the values of the listed columns to type string
    None and NaN values will also be converted to string

    Columns made by reshaping data (e.g. a pivot of incidents by area) may be absent
    for a period with no data in them; `skip_missing=True` leaves those out (and logs
    them) instead of raising a KeyError.

    Args:
        df (AnyDataFrame): The input DataFrame.
        columns (list[str]): List of columns to cast to string.
        skip_missing (bool): Skip listed columns that aren't in the table.

    Returns:
        AnyDataFrame: The modified DataFrame.

    Raises:
        KeyError: A listed column isn't in the table and `skip_missing` is False.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        if not skip_missing:
            raise KeyError(f"Columns not found: {missing}")
        logger.info("convert_column_values_to_string: skipped columns not in the table: %s", missing)

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return cast(AnyDataFrame, df)


@register()
def convert_values_to_timezone(
    df: AnyDataFrame,
    timezone: Annotated[str | datetime.tzinfo | TimezoneInfo, Field()],
    columns: Annotated[list[str], Field(description="The columns to convert.")],
    auto_detect: Annotated[
        bool,
        Field(
            description="Auto-detect all timezone-aware datetime columns to convert, ignoring the columns list.",
            exclude=True,
        ),
    ] = False,
) -> AnyDataFrame:
    """
    Converts the listed columns in the df to the timezone provided
    NOTE: Timezone naive timestamps are ignored
    Args:
        df (AnyDataFrame): The input DataFrame.
        timezone (str | datetime.tzinfo | TimezoneInfo): The timezone to convert to
        columns (list[str]): List of columns to cast to string.
        auto_detect (bool): If True, auto-detect all timezone-aware datetime columns,
            ignoring the columns list.

    Returns:
        AnyDataFrame: The modified DataFrame.
    """
    if isinstance(timezone, TimezoneInfo):
        timezone = timezone.utc_offset
    if auto_detect and columns:
        raise ValueError("Only one of auto_detect and columns may be passed.")
    if auto_detect:
        columns = [col for col in df.columns if isinstance(df[col].dtype, pd.DatetimeTZDtype)]
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
