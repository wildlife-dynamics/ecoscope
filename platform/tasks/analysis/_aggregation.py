from operator import add, floordiv, mod, mul, pow, sub, truediv
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame

ColumnName = Annotated[str, Field(description="Column to aggregate")]


@register()
def dataframe_count(
    df: AnyDataFrame,
) -> Annotated[int, Field(description="The number of rows in the DataFrame")]:
    return len(df)


@register()
def dataframe_column_mean(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[float, Field(description="The mean of the column")]:
    return df[column_name].mean()


@register()
def dataframe_column_sum(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[float, Field(description="The sum of the column")]:
    return df[column_name].sum()


@register()
def dataframe_column_max(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[float, Field(description="The max of the column")]:
    return df[column_name].max()


@register()
def dataframe_column_min(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[float, Field(description="The min of the column")]:
    return df[column_name].min()


@register()
def dataframe_column_nunique(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[int, Field(description="The number of unique values in the column")]:
    return df[column_name].nunique()


@register()
def dataframe_column_first_unique(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[int, Field(description="The first unique value in the column")]:
    return df[column_name].unique()[0]


@register()
def dataframe_column_percentile(
    df: AnyDataFrame,
    column_name: ColumnName,
    percentile: float,
) -> Annotated[
    int,
    Field(
        description="The percentile to calculate (e.g., 50 for median, 90 for 90th percentile)."
    ),
]:
    return np.nanpercentile(df[column_name].to_list(), percentile)


operations = {
    "add": add,
    "subtract": sub,
    "multiply": mul,
    "divide": truediv,
    "floor_divide": floordiv,
    "modulo": mod,
    "power": pow,
    "min": min,
    "max": max,
}

Operations = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "modulo",
    "power",
    "min",
    "max",
]


@register()
def apply_arithmetic_operation(
    a: Annotated[float | int, Field(description="The first number")],
    b: Annotated[float | int, Field(description="The second number")],
    operation: Annotated[
        Operations, Field(description="The arithmetic operation to apply")
    ],
) -> Annotated[
    float | int, Field(description="The result of the arithmetic operation")
]:
    return operations[operation](a, b)  # type: ignore[operator]


@register()
def apply_arithmetic_operation_over_rows(
    df: AnyDataFrame,
    column_a: Annotated[str, Field(description="The first column name")],
    column_b: Annotated[str, Field(description="The second column name")],
    output_column: Annotated[str, Field(description="The output column name")],
    operation: Annotated[
        Operations, Field(description="The arithmetic operation to apply")
    ],
) -> AnyDataFrame:
    df[output_column] = operations[operation](df[column_a], df[column_b])  # type: ignore[operator]
    return cast(AnyDataFrame, df)


@register()
def get_night_day_ratio(
    df: AnyGeoDataFrame,
) -> Annotated[float, Field(description="Night/Day ratio")]:
    from ecoscope.analysis import astronomy  # type: ignore[import-untyped]

    return astronomy.get_nightday_ratio(df)
