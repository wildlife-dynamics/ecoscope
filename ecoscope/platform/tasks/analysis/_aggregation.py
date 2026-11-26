from functools import reduce
from operator import add, floordiv, mod, mul, pow, sub, truediv
from typing import Annotated, Any, Callable, Literal, cast

import numpy as np
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
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
    Field(description="The percentile to calculate (e.g., 50 for median, 90 for 90th percentile)."),
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
    operation: Annotated[Operations, Field(description="The arithmetic operation to apply")],
) -> Annotated[float | int, Field(description="The result of the arithmetic operation")]:
    return operations[operation](a, b)  # type: ignore[operator]


# Row-wise versions: the built-in min/max can't compare two columns element by element
# ("The truth value of a Series is ambiguous"), so columns use numpy's.
row_operations = cast(dict[str, Callable[[Any, Any], Any]], {**operations, "min": np.minimum, "max": np.maximum})


@register()
def apply_arithmetic_operation_over_rows(
    df: AnyDataFrame,
    column_a: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The first column name. Use with column_b, or use columns instead."),
    ] = None,
    column_b: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The second column name. Use with column_a, or use columns instead."),
    ] = None,
    *,
    output_column: Annotated[str, Field(description="The output column name")],
    operation: Annotated[Operations, Field(description="The arithmetic operation to apply")],
    columns: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            description="Two or more columns, combined left to right, e.g. a + b + c. "
            "Use instead of column_a and column_b."
        ),
    ] = None,
) -> AnyDataFrame:
    """Apply an arithmetic operation across columns, row by row, into `output_column`.

    Give either `column_a` and `column_b`, or `columns` (two or more). Columns are
    combined left to right: for "subtract" `[a, b, c]` gives `(a - b) - c`, and
    "min" / "max" give the row-wise minimum / maximum.

    Raises:
        ValueError: Neither or both forms are given, or `columns` has fewer than two.
    """
    if columns is not None:
        if column_a is not None or column_b is not None:
            raise ValueError("Give either column_a and column_b, or columns, not both.")
        if len(columns) < 2:
            raise ValueError(f"columns needs at least two columns, got {columns!r}.")
        names = list(columns)
    elif column_a is not None and column_b is not None:
        names = [column_a, column_b]
    else:
        raise ValueError("Give column_a and column_b, or columns.")

    df[output_column] = reduce(row_operations[operation], (df[name] for name in names))
    return cast(AnyDataFrame, df)


@register()
def get_night_day_ratio(
    df: AnyGeoDataFrame,
) -> Annotated[float, Field(description="Night/Day ratio")]:
    from astropy.utils import iers  # type: ignore[import-untyped]

    from ecoscope.analysis import astronomy

    # See classify_is_night for rationale on disabling auto-download.
    with iers.conf.set_temp("auto_download", False):
        with iers.conf.set_temp("auto_max_age", None):
            with iers.conf.set_temp("iers_degraded_accuracy", "warn"):
                # get_nightday_ratio now also reports the number of days calculated in terms of
                # sunrise/sunsets. At the task level we report only the ratio for now.
                return astronomy.get_nightday_ratio(df).ratio
