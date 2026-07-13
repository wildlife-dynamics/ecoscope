from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from wt_registry import register

from ecoscope.platform.tasks.analysis._summary import (
    CoverageSummaryParam,
    NumericSummaryParam,
    SummaryParam,
    TallySummaryParam,
)
from ecoscope.platform.tasks.transformation._unit import Unit


# Thin patrol-aware wrappers over SummaryParam: each preset knows its column,
# statistic, and display name, so the form only asks for what varies (target
# unit, swath width). The `custom` variant is the escape hatch to the full
# SummaryParam fields.
class PatrolCountMetric(BaseModel):
    model_config = ConfigDict(title="Patrol Count")
    metric: Annotated[Literal["patrol_count"], Field(default="patrol_count", title="Metric")] = "patrol_count"

    def to_summary_param(self) -> TallySummaryParam:
        return TallySummaryParam(display_name="Patrol Count", aggregator="nunique", column="patrol_id")


class PatrolDaysMetric(BaseModel):
    model_config = ConfigDict(title="Patrol Days")
    metric: Annotated[Literal["patrol_days"], Field(default="patrol_days", title="Metric")] = "patrol_days"

    def to_summary_param(self) -> TallySummaryParam:
        return TallySummaryParam(display_name="Patrol Days", aggregator="nunique", column="segment_start_date")


class TotalDistanceMetric(BaseModel):
    model_config = ConfigDict(title="Total Distance")
    metric: Annotated[Literal["total_distance"], Field(default="total_distance", title="Metric")] = "total_distance"
    unit: Annotated[Literal["km", "m"], Field(default="km", title="Unit")] = "km"

    def to_summary_param(self) -> NumericSummaryParam:
        return NumericSummaryParam(
            display_name=f"Total Distance ({self.unit})",
            aggregator="sum",
            column="dist_meters",
            original_unit=Unit.METER,
            new_unit=Unit(self.unit),
            decimal_places=1,
        )


class TotalDurationMetric(BaseModel):
    model_config = ConfigDict(title="Total Duration")
    metric: Annotated[Literal["total_duration"], Field(default="total_duration", title="Metric")] = "total_duration"
    unit: Annotated[Literal["h", "d"], Field(default="h", title="Unit")] = "h"

    def to_summary_param(self) -> NumericSummaryParam:
        label = {"h": "hrs", "d": "days"}[self.unit]
        return NumericSummaryParam(
            display_name=f"Total Duration ({label})",
            aggregator="sum",
            column="timespan_seconds",
            original_unit=Unit.SECOND,
            new_unit=Unit(self.unit),
            decimal_places=1,
        )


class AreaCoveredMetric(BaseModel):
    model_config = ConfigDict(title="Area Covered")
    metric: Annotated[Literal["area_covered"], Field(default="area_covered", title="Metric")] = "area_covered"
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

    def to_summary_param(self) -> CoverageSummaryParam:
        prefix = "Merged" if self.merged else "Unmerged"
        return CoverageSummaryParam(
            display_name=f"{prefix} Area Covered (km²)",
            aggregator="coverage_area",
            merged=self.merged,
            swath_width_meters=self.swath_width_meters,
            decimal_places=1,
        )


class CustomMetric(BaseModel):
    model_config = ConfigDict(title="Custom")
    metric: Annotated[Literal["custom"], Field(default="custom", title="Metric")] = "custom"
    param: Annotated[SummaryParam, Field(title=" ", description="Full metric definition.")]

    def to_summary_param(self) -> SummaryParam:
        return self.param


PatrolSummaryMetric = Annotated[
    Union[
        PatrolCountMetric,
        PatrolDaysMetric,
        TotalDistanceMetric,
        TotalDurationMetric,
        AreaCoveredMetric,
        CustomMetric,
    ],
    Field(discriminator="metric"),
]

_PatrolSummaryMetricAdapter: TypeAdapter = TypeAdapter(PatrolSummaryMetric)

_DEFAULT_PATROL_SUMMARY_METRICS: list = [
    {"metric": "patrol_count"},
    {"metric": "total_distance", "unit": "km"},
    {"metric": "total_duration", "unit": "h"},
    {"metric": "patrol_days"},
    {"metric": "area_covered", "merged": True, "swath_width_meters": 500.0},
    {"metric": "area_covered", "merged": False, "swath_width_meters": 500.0},
]


@register()
def set_patrol_summary_metrics(
    metrics: Annotated[
        list[PatrolSummaryMetric],
        Field(
            default=_DEFAULT_PATROL_SUMMARY_METRICS,
            description="Metrics shown as columns in the patrol summary table. Add or remove rows to customize.",
        ),
    ] = _DEFAULT_PATROL_SUMMARY_METRICS,
) -> Annotated[list[SummaryParam], Field(description="Summary metric parameters")]:
    validated = [m if isinstance(m, BaseModel) else _PatrolSummaryMetricAdapter.validate_python(m) for m in metrics]
    return [m.to_summary_param() for m in validated]
