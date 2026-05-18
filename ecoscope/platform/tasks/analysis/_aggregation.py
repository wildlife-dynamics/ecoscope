import logging
from operator import add, floordiv, mod, mul, pow, sub, truediv
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame

logger = logging.getLogger(__name__)

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


@register()
def apply_arithmetic_operation_over_rows(
    df: AnyDataFrame,
    column_a: Annotated[str, Field(description="The first column name")],
    column_b: Annotated[str, Field(description="The second column name")],
    output_column: Annotated[str, Field(description="The output column name")],
    operation: Annotated[Operations, Field(description="The arithmetic operation to apply")],
) -> AnyDataFrame:
    df[output_column] = operations[operation](df[column_a], df[column_b])  # type: ignore[operator]
    return cast(AnyDataFrame, df)


@register()
def get_night_day_ratio(
    df: AnyGeoDataFrame,
) -> Annotated[float, Field(description="Night/Day ratio")]:
    # TEMP: stage timing instrumentation to diagnose Cloud Run slowness. Revert when done.
    import time

    t0 = time.perf_counter()
    from astropy.utils import iers

    t_iers_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    from ecoscope.analysis import astronomy

    t_astronomy_import = time.perf_counter() - t0

    # See classify_is_night for rationale — disable IERS auto-download to avoid
    # cold-start network IO on cloud workers.
    iers.conf.auto_download = False

    t0 = time.perf_counter()
    result = astronomy.get_nightday_ratio(df)
    t_ratio = time.perf_counter() - t0

    logger.warning(
        "get_night_day_ratio stage timings: import_iers=%.3fs import_astronomy=%.3fs "
        "get_nightday_ratio=%.3fs n=%d iers.auto_download=%s",
        t_iers_import,
        t_astronomy_import,
        t_ratio,
        len(df),
        iers.conf.auto_download,
    )

    return result
