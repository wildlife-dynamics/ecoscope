"""Aggregation, density, and summary analysis functions.

Use this module to compute statistics over DataFrame columns, calculate spatial
feature densities and time-density surfaces, and summarize rows with custom
aggregation operations.
"""

from ._aggregation import (
    apply_arithmetic_operation,
    apply_arithmetic_operation_over_rows,
    dataframe_column_first_unique,
    dataframe_column_max,
    dataframe_column_mean,
    dataframe_column_min,
    dataframe_column_nunique,
    dataframe_column_percentile,
    dataframe_column_sum,
    dataframe_count,
    get_night_day_ratio,
)
from ._calculate_feature_density import calculate_feature_density
from ._create_meshgrid import create_meshgrid
from ._summary import aggregate_over_rows, summarize_df
from ._time_density import (
    TimeDensityReturnGDFSchema,
    calculate_elliptical_time_density,
    calculate_linear_time_density,
)

__all__ = [
    "apply_arithmetic_operation",
    "apply_arithmetic_operation_over_rows",
    "dataframe_column_first_unique",
    "dataframe_column_max",
    "dataframe_column_mean",
    "dataframe_column_min",
    "dataframe_column_nunique",
    "dataframe_column_percentile",
    "dataframe_column_sum",
    "dataframe_count",
    "get_night_day_ratio",
    "calculate_feature_density",
    "create_meshgrid",
    "aggregate_over_rows",
    "summarize_df",
    "TimeDensityReturnGDFSchema",
    "calculate_elliptical_time_density",
    "calculate_linear_time_density",
]
