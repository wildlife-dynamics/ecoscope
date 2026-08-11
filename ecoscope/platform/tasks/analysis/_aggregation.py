from operator import add, floordiv, mod, mul, pow, sub, truediv
from typing import Annotated, Literal, TypeAlias, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame

ColumnName = Annotated[str, Field(description="Column to aggregate")]


class CountAggregation(BaseModel):
    """Count rows — no column needed."""

    model_config = ConfigDict(json_schema_extra={"title": "Count"})
    count_or_sum: Annotated[
        Literal["Count"],
        Field(default="Count", title="Aggregation"),
    ] = "Count"


class SumOfColumnAggregation(BaseModel):
    """Sum the values of a chosen column."""

    model_config = ConfigDict(json_schema_extra={"title": "Sum of Column"})
    count_or_sum: Annotated[
        Literal["Sum of Column"],
        Field(default="Sum of Column", title="Aggregation"),
    ] = "Sum of Column"
    column: Annotated[
        str,
        Field(
            default="",
            title="Column",
            description="Column whose values are summed instead of counting rows.",
        ),
    ] = ""


def make_aggregation_json_schema_extra(
    *,
    aggregation_title: str = "Aggregation",
    count_title: str = "Count",
    sum_title: str = "Sum of Column",
    column_title: str = "Column",
    column_description: str = "Column whose values are summed instead of counting rows.",
):
    """Flat allOf/if/then form schema for the Count | SumOfColumn union.

    RJSF discards branch formData when an anyOf selection changes, so a typed
    column would be lost on mode toggle. Rendering the union as a flat object
    with a conditionally revealed column keeps the value. No
    additionalProperties/unevaluatedProperties and no default on the
    conditional field — the only shape that renders, retains values, and
    passes 2020-12 submit validation. A retained "column" while Count is
    selected is ignored by CountAggregation validation.
    """

    def _flat_conditional_json_schema(schema: dict) -> None:
        schema.pop("anyOf", None)
        schema.update(
            {
                "type": "object",
                "title": "",
                "properties": {
                    "count_or_sum": {
                        "type": "string",
                        "title": aggregation_title,
                        "default": "Count",
                        "oneOf": [
                            {"const": "Count", "title": count_title},
                            {"const": "Sum of Column", "title": sum_title},
                        ],
                    },
                },
                "allOf": [
                    {
                        "if": {"properties": {"count_or_sum": {"const": "Sum of Column"}}},
                        "then": {
                            "properties": {
                                "column": {
                                    "type": "string",
                                    "title": column_title,
                                    "description": column_description,
                                },
                            },
                        },
                    },
                ],
            }
        )

    return _flat_conditional_json_schema


AggregationAnnotation: TypeAlias = Annotated[
    CountAggregation | SumOfColumnAggregation | SkipJsonSchema[None],
    Field(
        default=None,
        json_schema_extra=make_aggregation_json_schema_extra(),
    ),
]


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
    from astropy.utils import iers  # type: ignore[import-untyped]

    from ecoscope.analysis import astronomy

    # See classify_is_night for rationale on disabling auto-download.
    with iers.conf.set_temp("auto_download", False):
        with iers.conf.set_temp("auto_max_age", None):
            with iers.conf.set_temp("iers_degraded_accuracy", "warn"):
                return astronomy.get_nightday_ratio(df)
