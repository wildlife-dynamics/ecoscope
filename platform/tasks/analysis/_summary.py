from typing import Annotated, Literal, cast

import pandas as pd
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.tasks.analysis._aggregation import (
    get_night_day_ratio,
)

AggOperations = Literal[
    "sum",
    "count",
    "min",
    "max",
    "mean",
    "unique",
    "nunique",
    "median",
    "night_day_ratio",
]


class SummaryParam(BaseModel):
    display_name: str
    aggregator: AggOperations
    column: str | SkipJsonSchema[None] = None
    original_unit: Unit | SkipJsonSchema[None] = None
    new_unit: Unit | SkipJsonSchema[None] = None
    decimal_places: int | SkipJsonSchema[None] = 2

    @model_validator(mode="after")
    def check_column_and_unit(self):
        if self.aggregator != "night_day_ratio" and self.column is None:
            raise ValueError(
                "column cannot be None if aggregator is not night_day_ratio"
            )

        if (self.original_unit is None) != (self.new_unit is None):
            raise ValueError(
                "original_unit and new_unit must either both be None or both exist"
            )

        return self


@register()
def summarize_df(
    df: AnyDataFrame,
    summary_params: Annotated[
        list[SummaryParam],
        Field(
            description="The parameters that define how to calculate summary statistics.",
        ),
    ],
    groupby_cols: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=None,
            description="The columns to group by. If None, the summary is calculated for the entire DataFrame.",
        ),
    ] = None,
    reset_index: Annotated[
        bool | SkipJsonSchema[None],
        AdvancedField(
            default=False,
            description="Whether to reset the dataframe index after summarizing.",
        ),
    ] = False,
) -> Annotated[AnyDataFrame, Field(description="Summary Table")]:
    def summarize_column(df, param):
        result = 0
        if param.aggregator == "night_day_ratio":
            result = get_night_day_ratio(df)
        else:
            result = df[param.column].agg(param.aggregator)

        if param.original_unit and param.new_unit:
            result = with_unit(result, param.original_unit, param.new_unit).value

        if param.decimal_places:
            result = round(result, param.decimal_places)

        return result

    def summarize(df, summary_params):
        return pd.Series(
            {
                param.display_name: summarize_column(df, param)
                for param in summary_params
            }
        )

    if groupby_cols:
        result_df = df.groupby(groupby_cols).apply(  # type: ignore[call-overload]
            lambda x: summarize(x, summary_params), include_groups=False
        )
    else:
        series = summarize(df, summary_params)
        result_df = pd.DataFrame([series], columns=series.index)

    if reset_index:
        result_df.reset_index(drop=False, inplace=True)
    return cast(AnyDataFrame, result_df)


@register()
def aggregate_over_rows(
    df: AnyDataFrame,
    agg_ops: Annotated[
        AggOperations,
        Field(
            description="The parameters that define how to calculate summary statistics.",
        ),
    ],
    output_column: Annotated[
        str,
        Field(
            description="The output column name.",
        ),
    ],
    columns: Annotated[
        list[str],
        Field(
            description="The list of columns.",
        ),
    ],
) -> AnyDataFrame:
    df[output_column] = df[columns].agg(func=[agg_ops], axis=1)
    return cast(AnyDataFrame, df)
