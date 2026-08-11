from typing import Annotated, List, Literal, cast

import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame

TemporalComponent = Literal[
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "date",
    "time",
    "timetz",
    "dayofyear",
    "weekofyear",
    "week",
    "dayofweek",
    "weekday",
    "quarter",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
    "is_leap_year",
    "month_name",
    "day_name",
]


@register()
def decompose_datetime(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="Input DataFrame containing the datetime column.",
            exclude=True,
        ),
    ],
    datetime_column: Annotated[str, Field(description="Name of the datetime column to decompose.")],
    components: Annotated[
        List[TemporalComponent],
        Field(description="List of temporal components to extract (e.g., 'year', 'month')."),
    ],
    remove_source: Annotated[
        bool,
        Field(description="If True, drops the original datetime column after extraction."),
    ] = False,
    column_prefix: Annotated[
        str | None,
        Field(description="Custom prefix for generated columns (defaults to datetime_column name)."),
    ] = None,
) -> AnyDataFrame:
    """Decompose a datetime column into its constituent temporal components.

    Extracts specified datetime attributes using the pandas datetime accessor,
    creating a `<prefix>_<component>` column for each component while
    optionally preserving or removing the original datetime column.
    """
    if datetime_column not in df.columns:
        raise KeyError(f"Column '{datetime_column}' not found in DataFrame. Available columns: {list(df.columns)}")

    result = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result[datetime_column]):
        result[datetime_column] = pd.to_datetime(result[datetime_column])

    prefix = column_prefix or datetime_column
    dt_accessor = result[datetime_column].dt

    for component in components:
        if not hasattr(dt_accessor, component):
            raise AttributeError(
                f"Invalid temporal component: '{component}'. "
                f"Pandas datetime accessor does not support this attribute."
            )

        attribute = getattr(dt_accessor, component)
        value = attribute() if callable(attribute) else attribute
        result[f"{prefix}_{component}"] = value

    if remove_source:
        result = result.drop(columns=[datetime_column])

    return cast(AnyDataFrame, result)
