from typing import Annotated, Literal, Union, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._aggregation import (
    get_night_day_ratio,
)
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit

# Plain pandas aggregations, applied via df.agg. The extra summarize_df-only
# aggregators (night_day_ratio, coverage_area) live on their own SummaryParam
# variants below.
AggOperations = Literal[
    "sum",
    "count",
    "min",
    "max",
    "mean",
    "nunique",
    "median",
]


def _coverage_area_km2(
    gdf: AnyGeoDataFrame,
    swath_width_meters: float,
    merged: bool,
    area_crs: str = "EPSG:6933",
) -> float:
    """
    Compute the ground area (km²) covered by buffering track segments.

    Each segment is buffered by ``swath_width_meters / 2`` per side to form a
    corridor of full width ``swath_width_meters``. When ``merged`` is True the
    area of the union of all corridors is returned (overlaps counted once);
    otherwise the per-segment corridor areas are summed.
    """
    if gdf.empty:
        return 0.0

    buffers = gdf.to_crs(area_crs).geometry.buffer(swath_width_meters / 2)  # type: ignore[operator]

    if merged:
        area_m2 = buffers.union_all().area
    else:
        area_m2 = buffers.area.sum()

    return area_m2 / 1e6


# `SummaryParam` is a discriminated union (on `aggregator`) so that each metric
# type exposes only the fields that apply to it: count-style metrics need just a
# column; numeric metrics add unit conversion + rounding; coverage metrics take a
# swath width and have no column; the night/day ratio has neither column nor units.
class _BaseSummaryParam(BaseModel):
    display_name: Annotated[str, Field(description="Column header shown in the summary table.")]


class TallySummaryParam(_BaseSummaryParam):
    """Count-style metric (count / nunique) over a single column."""

    model_config = ConfigDict(title="Count Metric")
    aggregator: Annotated[Literal["count", "nunique"], Field(title="Statistic")]
    column: Annotated[str, Field(description="Column to aggregate.")]


class NumericSummaryParam(_BaseSummaryParam):
    """Numeric reduction (sum / min / max / mean / median) with optional unit conversion."""

    model_config = ConfigDict(title="Numeric Metric")
    aggregator: Annotated[Literal["sum", "min", "max", "mean", "median"], Field(title="Statistic")]
    column: Annotated[str, Field(description="Column to aggregate.")]
    original_unit: Unit | SkipJsonSchema[None] = None
    new_unit: Unit | SkipJsonSchema[None] = None
    decimal_places: int | SkipJsonSchema[None] = 2

    @model_validator(mode="after")
    def check_unit(self):
        if (self.original_unit is None) != (self.new_unit is None):
            raise ValueError("original_unit and new_unit must either both be None or both exist")
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


SummaryParam = Annotated[
    Union[
        TallySummaryParam,
        NumericSummaryParam,
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
            result = _coverage_area_km2(df, param.swath_width_meters, merged=param.merged)
        else:
            result = df[param.column].agg(param.aggregator)

        original_unit = getattr(param, "original_unit", None)
        new_unit = getattr(param, "new_unit", None)
        if original_unit and new_unit:
            result = with_unit(result, original_unit, new_unit).value

        decimal_places = getattr(param, "decimal_places", None)
        if decimal_places:
            result = round(result, decimal_places)

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
