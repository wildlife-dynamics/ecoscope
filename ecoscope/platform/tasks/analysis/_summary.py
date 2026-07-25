from typing import Annotated, Literal, Union, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.base.utils import coverage_area_km2
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from ecoscope.platform.tasks.analysis._aggregation import (
    get_night_day_ratio,
)
from ecoscope.platform.tasks.transformation._unit import UNIT_OPTIONS, Unit, with_unit

AggOperations = Literal[
    "sum",
    "count",
    "min",
    "max",
    "mean",
    "nunique",
    "median",
]


class _BaseSummaryParam(BaseModel):
    display_name: Annotated[str, Field(title="Display Name", description="Column header shown in the summary table.")]


# The unit fields are excluded from the model's own JSON schema (SkipJsonSchema)
# and re-introduced via the allOf/if/then block in json_schema_extra, so the form
# only shows them once "Convert Units" is checked. if/then (not the legacy
# `dependencies` keyword) is understood by BOTH the RJSF renderer and JSON Schema
# 2020-12 validators (ecoscope-server validates submitted formdata with
# Draft202012Validator, which silently ignores `dependencies`). See
# wildlife-dynamics/wt-download-subject-tracks#31 for the full dialect story.
# Note the docstring renders as the form's helper text.
class StatSummaryParam(_BaseSummaryParam):
    """Pick a column and statistic, with optional unit conversion."""

    model_config = ConfigDict(
        title="Statistic",
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"convert_units": {"const": True}}},
                    "then": {
                        "properties": {
                            # No `default` on these: a default on a conditional branch
                            # gets seeded into formData even while unchecked, leaving
                            # an orphaned value behind.
                            "original_unit": {
                                "title": "Original Unit",
                                "type": "string",
                                "oneOf": UNIT_OPTIONS,
                            },
                            "new_unit": {
                                "title": "New Unit",
                                "type": "string",
                                "oneOf": UNIT_OPTIONS,
                            },
                        }
                    },
                }
            ]
        },
    )
    aggregator: Annotated[AggOperations, Field(title="Statistic")]
    column: Annotated[str, Field(title="Column", description="Column to aggregate.")]
    decimal_places: Annotated[int, Field(default=2, title="Decimal Places")] = 2
    convert_units: Annotated[bool, Field(default=False, title="Convert Units")] = False
    original_unit: SkipJsonSchema[Unit | None] = None
    new_unit: SkipJsonSchema[Unit | None] = None

    @model_validator(mode="before")
    @classmethod
    def infer_convert_units(cls, data):
        # Specs written before the Convert Units checkbox pass units directly;
        # honor them unless the box is explicitly unchecked.
        if (
            isinstance(data, dict)
            and "convert_units" not in data
            and (data.get("original_unit") is not None or data.get("new_unit") is not None)
        ):
            data["convert_units"] = True
        return data

    @model_validator(mode="after")
    def check_units(self):
        if self.convert_units and (self.original_unit is None or self.new_unit is None):
            raise ValueError("select both an original and a new unit when unit conversion is enabled")
        if not self.convert_units:
            # units left behind by unchecking "Convert Units" in the form are ignored
            self.original_unit = None
            self.new_unit = None
        return self


class CoverageSummaryParam(_BaseSummaryParam):
    """Ground area (km²) covered by buffering track segments. Has no column."""

    model_config = ConfigDict(title="Area Covered Metric")
    aggregator: Annotated[Literal["coverage_area"], Field(title="Statistic")]
    merged: Annotated[
        bool,
        Field(
            default=True,
            title="Merged",
            description="Merge overlapping swaths before measuring (union); otherwise sum per-segment areas.",
        ),
    ] = True
    swath_width_meters: Annotated[
        float,
        Field(default=500.0, title="Swath Width (m)", description="Full corridor width in meters."),
    ] = 500.0
    decimal_places: int | SkipJsonSchema[None] = 2


class NightDayRatioSummaryParam(_BaseSummaryParam):
    """Ratio of night to day fixes. Has no column or units."""

    model_config = ConfigDict(title="Night/Day Ratio Metric")
    aggregator: Annotated[Literal["night_day_ratio"], Field(title="Statistic")]
    decimal_places: int | SkipJsonSchema[None] = 2


class UniqueSummaryParam(_BaseSummaryParam):
    """The distinct values of a column, shown as a comma-separated list."""

    model_config = ConfigDict(title="Unique Values")
    aggregator: Annotated[Literal["unique"], Field(title="Statistic")]
    column: Annotated[str, Field(title="Column", description="Column to aggregate.")]
    decimal_places: SkipJsonSchema[None] = None


SummaryParam = Annotated[
    Union[
        StatSummaryParam,
        UniqueSummaryParam,
        CoverageSummaryParam,
        NightDayRatioSummaryParam,
    ],
    Field(discriminator="aggregator"),
]


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
        elif param.aggregator == "coverage_area":
            result = coverage_area_km2(df, param.swath_width_meters, merged=param.merged)
        elif param.aggregator == "unique":
            result = ", ".join(str(value) for value in df[param.column].unique())
        else:
            result = df[param.column].agg(param.aggregator)

        if isinstance(param, StatSummaryParam) and param.original_unit and param.new_unit:
            result = with_unit(result, param.original_unit, param.new_unit).value

        if param.decimal_places is not None:
            result = round(result, param.decimal_places)

        return result

    def summarize(df, summary_params):
        return pd.Series({param.display_name: summarize_column(df, param) for param in summary_params})

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
