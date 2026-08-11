from collections.abc import Sequence
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
)
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.tasks.analysis._aggregation import (
    CountAggregation,
    SumOfColumnAggregation,
    make_aggregation_json_schema_extra,
)
from ecoscope.platform.tasks.analysis._summary import (
    CoverageSummaryParam,
    RatioOperand,
    RatioSummaryParam,
    StatSummaryParam,
    SummaryParam,
)
from ecoscope.platform.tasks.transformation._unit import Unit, labeled_units


# Thin patrol-aware wrappers over SummaryParam: each preset knows its column,
# statistic, and display name, so the form only asks for what varies (target
# unit, swath width). The `custom` variant is the escape hatch to the full
# SummaryParam fields.
class PatrolCountMetric(BaseModel):
    model_config = ConfigDict(title="Patrol Count")
    metric: Annotated[Literal["patrol_count"], Field(default="patrol_count", title="Metric")] = "patrol_count"

    def to_summary_param(self) -> StatSummaryParam:
        return StatSummaryParam(display_name="Patrol Count", aggregator="nunique", column="patrol_id")


class PatrolDaysMetric(BaseModel):
    model_config = ConfigDict(title="Patrol Days")
    metric: Annotated[Literal["patrol_days"], Field(default="patrol_days", title="Metric")] = "patrol_days"

    def to_summary_param(self) -> StatSummaryParam:
        return StatSummaryParam(
            display_name="Patrol Days",
            aggregator="nunique",
            column="segment_start_date",
        )


class TotalDistanceMetric(BaseModel):
    model_config = ConfigDict(title="Total Distance")
    metric: Annotated[Literal["total_distance"], Field(default="total_distance", title="Metric")] = "total_distance"
    unit: Annotated[
        Literal["km", "m"],
        Field(default="km", title="Unit", json_schema_extra=labeled_units(Unit.KILOMETER, Unit.METER)),
    ] = "km"

    def to_summary_param(self) -> StatSummaryParam:
        return StatSummaryParam(
            display_name=f"Total Distance ({self.unit})",
            aggregator="sum",
            column="dist_meters",
            convert_units=True,
            original_unit=Unit.METER,
            new_unit=Unit(self.unit),
            decimal_places=2,
        )


class TotalDurationMetric(BaseModel):
    model_config = ConfigDict(title="Total Duration")
    metric: Annotated[Literal["total_duration"], Field(default="total_duration", title="Metric")] = "total_duration"
    unit: Annotated[
        Literal["h", "d"],
        Field(default="h", title="Unit", json_schema_extra=labeled_units(Unit.HOUR, Unit.DAY)),
    ] = "h"

    def to_summary_param(self) -> StatSummaryParam:
        label = {"h": "hrs", "d": "days"}[self.unit]
        return StatSummaryParam(
            display_name=f"Total Duration ({label})",
            aggregator="sum",
            column="timespan_seconds",
            convert_units=True,
            original_unit=Unit.SECOND,
            new_unit=Unit(self.unit),
            decimal_places=2,
        )


class MergedAreaCoveredMetric(BaseModel):
    """Area covered with overlaps counted once — the distinct ground footprint."""

    model_config = ConfigDict(title="Area Covered (Merged)")
    metric: Annotated[
        Literal["area_covered_merged"],
        Field(default="area_covered_merged", title="Metric"),
    ] = "area_covered_merged"
    swath_width_meters: Annotated[
        float,
        Field(
            default=500.0,
            title="Swath Width (m)",
            description="Full corridor width in meters.",
        ),
    ] = 500.0

    def to_summary_param(self) -> CoverageSummaryParam:
        return CoverageSummaryParam(
            display_name="Merged Area Covered (km²)",
            aggregator="coverage_area",
            merged=True,
            swath_width_meters=self.swath_width_meters,
            decimal_places=2,
        )


class UnmergedAreaCoveredMetric(BaseModel):
    """Area covered summed segment by segment — total patrol efforts."""

    model_config = ConfigDict(title="Area Covered (Unmerged)")
    metric: Annotated[
        Literal["area_covered_unmerged"],
        Field(default="area_covered_unmerged", title="Metric"),
    ] = "area_covered_unmerged"
    swath_width_meters: Annotated[
        float,
        Field(
            default=500.0,
            title="Swath Width (m)",
            description="Full corridor width in meters.",
        ),
    ] = 500.0

    def to_summary_param(self) -> CoverageSummaryParam:
        return CoverageSummaryParam(
            display_name="Unmerged Area Covered (km²)",
            aggregator="coverage_area",
            merged=False,
            swath_width_meters=self.swath_width_meters,
            decimal_places=2,
        )


# Inherits the full statistic form (column, statistic, units-behind-a-checkbox
# `dependencies` block) from StatSummaryParam; only adds the `metric`
# discriminator for the patrol preset union.
class CustomMetric(StatSummaryParam):
    """Define your own metric: pick a column and statistic, with optional unit conversion."""

    model_config = ConfigDict(title="Custom")
    metric: Annotated[Literal["custom"], Field(default="custom", title="Metric")] = "custom"

    def to_summary_param(self) -> StatSummaryParam:
        return StatSummaryParam(**self.model_dump(exclude={"metric"}))


PatrolSummaryMetric = Annotated[
    Union[
        PatrolCountMetric,
        PatrolDaysMetric,
        TotalDistanceMetric,
        TotalDurationMetric,
        MergedAreaCoveredMetric,
        UnmergedAreaCoveredMetric,
        CustomMetric,
    ],
    Field(discriminator="metric"),
]

_PatrolSummaryMetricAdapter: TypeAdapter = TypeAdapter(PatrolSummaryMetric)

_DEFAULT_PATROL_SUMMARY_METRICS: tuple = (
    {"metric": "patrol_count"},
    {"metric": "total_distance", "unit": "km"},
    {"metric": "total_duration", "unit": "h"},
    {"metric": "patrol_days"},
    {"metric": "area_covered_merged", "swath_width_meters": 500.0},
    {"metric": "area_covered_unmerged", "swath_width_meters": 500.0},
)


@register()
def set_patrol_summary_metrics(
    metrics: Annotated[
        Sequence[PatrolSummaryMetric],
        Field(
            default=_DEFAULT_PATROL_SUMMARY_METRICS,
            description="Metrics shown as columns in the patrol summary table. Add or remove rows to customize.",
        ),
    ] = _DEFAULT_PATROL_SUMMARY_METRICS,
) -> Annotated[list[SummaryParam], Field(description="Summary metric parameters")]:
    validated = [m if isinstance(m, BaseModel) else _PatrolSummaryMetricAdapter.validate_python(m) for m in metrics]
    return [m.to_summary_param() for m in validated]


def _event_numerator(aggregate_column: str | None) -> RatioOperand:
    # With no aggregate column, each event row counts as 1. With one, we sum it
    # over event rows; segment rows are NaN there, so pandas sum skips them.
    if aggregate_column is None:
        return RatioOperand(aggregator="count", column="event_type")
    return RatioOperand(aggregator="sum", column=aggregate_column)


_MINOR_WORDS = {"a", "an", "and", "of", "or", "per", "the"}


def _event_label(aggregate_column: str | None) -> str:
    # Column-less metrics count event rows: "Events per X" rates, "Total
    # Events" for the plain total (the Total prefix comes from the caller).
    if aggregate_column is None:
        return "Events"
    words = aggregate_column.replace("_", " ").lower().split()
    return " ".join(word if i > 0 and word in _MINOR_WORDS else word.capitalize() for i, word in enumerate(words))


# Encounter-rate presets, sharing the patrol presets above where the metric is
# the same (patrol count, total distance, total duration). The "Encounter ..."
# metrics follow the task-level aggregation ("Measure Encounters By"): they
# count event records in Number of Events mode and sum an event field over
# event rows in Sum of an Event Field mode — their own aggregate_column if
# set, else the task-level field — falling back to counting, and to
# "Events per ..." / "Total Events" display names, when neither names a
# field. Total Events always counts, regardless of the aggregation; both can
# appear in one table.
class TotalEventsMetric(BaseModel):
    model_config = ConfigDict(title="Total Events")
    metric: Annotated[Literal["total_events"], Field(default="total_events", title="Metric")] = "total_events"

    def to_summary_param(self) -> StatSummaryParam:
        return StatSummaryParam(display_name="Total Events", aggregator="count", column="event_type")


_PER_METRIC_AGGREGATE_COLUMN_FIELD = Field(
    default="",
    title="Event Field to Sum",
    description=(
        "Event details field whose values are totaled for this metric, using the"
        ' field title shown in EarthRanger (for example "Number of Animals").'
        " Leave blank to use the task-level Event Field to Sum."
    ),
)


class TotalEncounterMetric(BaseModel):
    model_config = ConfigDict(title="Total Encounter")
    metric: Annotated[
        Literal["total_encounter"],
        Field(default="total_encounter", title="Metric"),
    ] = "total_encounter"
    aggregate_column: Annotated[str, _PER_METRIC_AGGREGATE_COLUMN_FIELD] = ""

    def to_summary_param(self, aggregate_column: str | None = None) -> StatSummaryParam:
        column = self.aggregate_column or aggregate_column
        operand = _event_numerator(column)
        return StatSummaryParam(
            display_name=f"Total {_event_label(column)}",
            aggregator=operand.aggregator,
            column=operand.column,
        )


_UNIT_RATE_LABELS = {"h": "Hour", "d": "Day", "km": "Km", "m": "Meter"}
_PER_TIME_SCALES = {"h": 3600.0, "d": 86400.0}
_PER_DISTANCE_SCALES = {"km": 1000.0, "m": 1.0}


class EncountersPerDurationMetric(BaseModel):
    model_config = ConfigDict(title="Encounters per Duration")
    metric: Annotated[
        Literal["encounters_per_duration"],
        Field(default="encounters_per_duration", title="Metric"),
    ] = "encounters_per_duration"
    unit: Annotated[
        Literal["h", "d"],
        Field(default="h", title="Unit", json_schema_extra=labeled_units(Unit.HOUR, Unit.DAY)),
    ] = "h"
    aggregate_column: Annotated[str, _PER_METRIC_AGGREGATE_COLUMN_FIELD] = ""

    def to_summary_param(self, aggregate_column: str | None = None) -> RatioSummaryParam:
        column = self.aggregate_column or aggregate_column
        return RatioSummaryParam(
            display_name=f"{_event_label(column)} per {_UNIT_RATE_LABELS[self.unit]}",
            aggregator="ratio",
            numerator=_event_numerator(column),
            denominator=RatioOperand(aggregator="sum", column="timespan_seconds"),
            scale=_PER_TIME_SCALES[self.unit],
            decimal_places=2,
        )


class EncountersPerDistanceMetric(BaseModel):
    model_config = ConfigDict(title="Encounters per Distance")
    metric: Annotated[
        Literal["encounters_per_distance"],
        Field(default="encounters_per_distance", title="Metric"),
    ] = "encounters_per_distance"
    unit: Annotated[
        Literal["km", "m"],
        Field(default="km", title="Unit", json_schema_extra=labeled_units(Unit.KILOMETER, Unit.METER)),
    ] = "km"
    aggregate_column: Annotated[str, _PER_METRIC_AGGREGATE_COLUMN_FIELD] = ""

    def to_summary_param(self, aggregate_column: str | None = None) -> RatioSummaryParam:
        column = self.aggregate_column or aggregate_column
        return RatioSummaryParam(
            display_name=f"{_event_label(column)} per {_UNIT_RATE_LABELS[self.unit]}",
            aggregator="ratio",
            numerator=_event_numerator(column),
            denominator=RatioOperand(aggregator="sum", column="dist_meters"),
            scale=_PER_DISTANCE_SCALES[self.unit],
            decimal_places=2,
        )


class EncountersPerPatrolMetric(BaseModel):
    model_config = ConfigDict(title="Encounters per Patrol")
    metric: Annotated[
        Literal["encounters_per_patrol"],
        Field(default="encounters_per_patrol", title="Metric"),
    ] = "encounters_per_patrol"
    aggregate_column: Annotated[str, _PER_METRIC_AGGREGATE_COLUMN_FIELD] = ""

    def to_summary_param(self, aggregate_column: str | None = None) -> RatioSummaryParam:
        column = self.aggregate_column or aggregate_column
        return RatioSummaryParam(
            display_name=f"{_event_label(column)} per Patrol",
            aggregator="ratio",
            numerator=_event_numerator(column),
            denominator=RatioOperand(aggregator="nunique", column="patrol_id"),
            scale=1.0,
            decimal_places=2,
        )


class EncountersPerPatrolDayMetric(BaseModel):
    """Rate against patrol days — distinct calendar days with patrol movement
    (nunique segment_start_date, matching PatrolDaysMetric)."""

    model_config = ConfigDict(title="Encounters per Patrol Day")
    metric: Annotated[
        Literal["encounters_per_patrol_day"],
        Field(default="encounters_per_patrol_day", title="Metric"),
    ] = "encounters_per_patrol_day"
    aggregate_column: Annotated[str, _PER_METRIC_AGGREGATE_COLUMN_FIELD] = ""

    def to_summary_param(self, aggregate_column: str | None = None) -> RatioSummaryParam:
        column = self.aggregate_column or aggregate_column
        return RatioSummaryParam(
            display_name=f"{_event_label(column)} per Patrol Day",
            aggregator="ratio",
            numerator=_event_numerator(column),
            denominator=RatioOperand(aggregator="nunique", column="segment_start_date"),
            scale=1.0,
            decimal_places=2,
        )


# Union order drives the RJSF dropdown: keep it ALPHABETICAL by title.
# NOTE: the area-covered metrics buffer every geometry in the group — on the
# combined segments+events frame the event points add small discs.
EncounterRateMetric = Annotated[
    Union[
        MergedAreaCoveredMetric,  # Area Covered (Merged)
        UnmergedAreaCoveredMetric,  # Area Covered (Unmerged)
        CustomMetric,  # Custom
        EncountersPerDistanceMetric,  # Encounters per Distance
        EncountersPerDurationMetric,  # Encounters per Duration
        EncountersPerPatrolMetric,  # Encounters per Patrol
        EncountersPerPatrolDayMetric,  # Encounters per Patrol Day
        PatrolCountMetric,  # Patrol Count
        PatrolDaysMetric,  # Patrol Days
        TotalDistanceMetric,  # Total Distance
        TotalDurationMetric,  # Total Duration
        TotalEncounterMetric,  # Total Encounter
        TotalEventsMetric,  # Total Events
    ],
    Field(discriminator="metric"),
]

_EncounterRateMetricAdapter: TypeAdapter = TypeAdapter(EncounterRateMetric)

# The metrics that receive the task-level aggregate_column.
_EVENT_METRIC_TYPES = (
    TotalEncounterMetric,
    EncountersPerDurationMetric,
    EncountersPerDistanceMetric,
    EncountersPerPatrolMetric,
    EncountersPerPatrolDayMetric,
)

_DEFAULT_ENCOUNTER_RATE_METRICS: tuple = (
    {"metric": "total_encounter"},
    {"metric": "total_duration", "unit": "h"},
    {"metric": "encounters_per_duration", "unit": "h"},
    {"metric": "total_distance", "unit": "km"},
    {"metric": "encounters_per_distance", "unit": "km"},
)


# Event-relevant wording for this card (mirrors the encounter-rate map's
# aggregation field); the platform mechanism stays generic.
_EVENT_AGGREGATION_EXTRA = make_aggregation_json_schema_extra(
    aggregation_title="Measure Encounters By",
    count_title="Number of Events",
    sum_title="Sum of an Event Field",
    column_title="Event Field to Sum",
    column_description=(
        "Event details field whose values are totaled for the Encounter metrics"
        " (Total Encounter and the Encounters per Duration/Distance/Patrol"
        " rates), using the field title shown in EarthRanger (for example"
        ' "Number of Animals"). Total Events always counts events.'
    ),
)

_AggregationAdapter: TypeAdapter = TypeAdapter(CountAggregation | SumOfColumnAggregation)


@register()
def set_encounter_rate_metrics(
    aggregation: Annotated[
        CountAggregation | SumOfColumnAggregation | SkipJsonSchema[None],
        Field(default=None, json_schema_extra=_EVENT_AGGREGATION_EXTRA),
    ] = None,
    metrics: Annotated[
        Sequence[EncounterRateMetric],
        Field(
            default=_DEFAULT_ENCOUNTER_RATE_METRICS,
            description="Metrics shown as columns in the encounter rate table. Add or remove rows to customize.",
        ),
    ] = _DEFAULT_ENCOUNTER_RATE_METRICS,
) -> Annotated[list[SummaryParam], Field(description="Summary metric parameters")]:
    if isinstance(aggregation, dict):
        aggregation = _AggregationAdapter.validate_python(aggregation)
    # Empty column in sum mode degrades to counting, like the map task.
    aggregate_column = (
        aggregation.column if isinstance(aggregation, SumOfColumnAggregation) and aggregation.column else None
    )
    validated = [m if isinstance(m, BaseModel) else _EncounterRateMetricAdapter.validate_python(m) for m in metrics]
    return [
        m.to_summary_param(aggregate_column) if isinstance(m, _EVENT_METRIC_TYPES) else m.to_summary_param()
        for m in validated
    ]
