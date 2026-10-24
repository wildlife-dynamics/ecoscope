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
    decimal_places: Annotated[int | SkipJsonSchema[None], Field(default=2, title="Decimal Places")] = 2
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


class RatioOperand(BaseModel):
    aggregator: Annotated[AggOperations, Field(title="Statistic")]
    column: Annotated[str, Field(title="Column", description="Column to aggregate.")]


class RatioSummaryParam(_BaseSummaryParam):
    """Ratio of two aggregations per group, e.g. events per hour. NA when the denominator is 0."""

    model_config = ConfigDict(title="Ratio")
    aggregator: Annotated[Literal["ratio"], Field(title="Statistic")]
    numerator: Annotated[RatioOperand, Field(title="Numerator")]
    denominator: Annotated[RatioOperand, Field(title="Denominator")]
    scale: Annotated[
        float,
        Field(
            default=1.0,
            title="Scale",
            description="Multiplier applied to the ratio, e.g. 3600 to turn per-second into per-hour.",
        ),
    ] = 1.0
    decimal_places: Annotated[int | SkipJsonSchema[None], Field(default=2, title="Decimal Places")] = 2


SummaryParam = Annotated[
    Union[
        StatSummaryParam,
        UniqueSummaryParam,
        CoverageSummaryParam,
        NightDayRatioSummaryParam,
        RatioSummaryParam,
    ],
    Field(discriminator="aggregator"),
]


def summarize_column(df, param):
    result = 0
    if param.aggregator == "night_day_ratio":
        result = get_night_day_ratio(df)
    elif param.aggregator == "coverage_area":
        result = coverage_area_km2(df, param.swath_width_meters, merged=param.merged)
    elif param.aggregator == "unique":
        result = ", ".join(str(value) for value in df[param.column].unique())
    elif param.aggregator == "ratio":
        numerator = df[param.numerator.column].agg(param.numerator.aggregator)
        denominator = df[param.denominator.column].agg(param.denominator.aggregator)
        result = float("nan") if denominator == 0 else numerator / denominator * param.scale
    elif isinstance(param, StatSummaryParam) and param.aggregator == "sum" and param.column == "leg_member_days":
        # all-NaN means no member data at this site: report blank, not 0
        result = df[param.column].sum(min_count=1)
    else:
        result = df[param.column].agg(param.aggregator)

    if isinstance(param, StatSummaryParam) and param.original_unit and param.new_unit:
        result = with_unit(result, param.original_unit, param.new_unit).value

    if param.decimal_places is not None:
        result = round(result, param.decimal_places)

    return result


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


LEG_METRIC_COLUMNS: frozenset[str] = frozenset(
    {"leg_active_seconds", "patrol_elapsed_seconds", "patrol_segment_id", "leg_member_days"}
)


@register()
def select_track_metrics(
    summary_params: Annotated[list[SummaryParam], Field(description="All summary parameters.")],
) -> Annotated[list[SummaryParam], Field(description="Parameters computed over the trajectory frame.")]:
    """Keep the summary parameters computed over the trajectory frame (non-leg metrics)."""
    return [p for p in summary_params if getattr(p, "column", None) not in LEG_METRIC_COLUMNS]


@register()
def select_leg_metrics(
    summary_params: Annotated[list[SummaryParam], Field(description="All summary parameters.")],
) -> Annotated[list[SummaryParam], Field(description="Parameters computed over the leg frame.")]:
    """Keep the summary parameters computed over the leg frame: Active Duration,
    Elapsed Duration, Leg Count and Member-days.
    """
    return [p for p in summary_params if getattr(p, "column", None) in LEG_METRIC_COLUMNS]


@register()
def join_summary_tables(
    tables: Annotated[
        list[AnyDataFrame],
        Field(description="Summary tables to join, group-key summaries of the same groups (e.g. track then leg)."),
    ],
) -> Annotated[AnyDataFrame, Field(description="Joined summary table")]:
    """Outer-join summary tables on the group key, left to right. Empty tables are ignored,
    so existing track metrics never change; a single table is returned as is, and an empty
    list gives an empty frame. Taking a list lets a `groupbykey` pair with one side missing
    (a group key present on only one frame) still render.

    A table that adds no column of its own (every column it has is also on another table,
    e.g. a leg-side summary when no leg metric was selected, which carries only the group
    keys) is ignored too: it has nothing to contribute, and its keys would otherwise become
    all-blank rows in the outer join. When every table is key-only the first is kept.

    Two modes per join:
    - Both frames have a plain `RangeIndex` (`summarize_df` ran with `reset_index=True`):
      outer-merge on the columns the two frames share, which are the group-key columns.
    - Otherwise (or when no column is shared, e.g. two ungrouped one-row summaries):
      outer-join on the index (the group key `summarize_df` left there).
    """
    present = [t for t in tables if not t.empty]
    if not present:
        return cast(AnyDataFrame, pd.DataFrame())
    contributing = [
        t
        for i, t in enumerate(present)
        if not set(t.columns) <= {c for j, o in enumerate(present) if j != i for c in o.columns}
    ]
    present = contributing or present[:1]
    out = present[0]
    for right in present[1:]:
        if isinstance(out.index, pd.RangeIndex) and isinstance(right.index, pd.RangeIndex):
            shared = [c for c in out.columns if c in right.columns]
            if shared:
                out = out.merge(right, how="outer", on=shared)
                continue
        out = out.join(right, how="outer")
    return cast(AnyDataFrame, out)


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
