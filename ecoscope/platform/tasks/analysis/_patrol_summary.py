from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
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
        return TallySummaryParam(
            display_name="Patrol Days",
            aggregator="nunique",
            column="segment_start_date",
        )


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
            decimal_places=1,
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
            decimal_places=1,
        )


# Labeled select options for the unit fields on the custom metric's
# unit-conversion branch (oneOf const/title pairs render as a labeled dropdown).
_UNIT_OPTIONS: list[JsonValue] = [
    {"const": "m", "title": "Meters (m)"},
    {"const": "km", "title": "Kilometers (km)"},
    {"const": "m²", "title": "Square Meters (m²)"},
    {"const": "km²", "title": "Square Kilometers (km²)"},
    {"const": "s", "title": "Seconds (s)"},
    {"const": "h", "title": "Hours (h)"},
    {"const": "d", "title": "Days (d)"},
    {"const": "m/s", "title": "Meters per Second (m/s)"},
    {"const": "km/h", "title": "Kilometers per Hour (km/h)"},
]

CustomStatistic = Literal["count", "nunique", "sum", "min", "max", "mean", "median"]


class CustomMetric(BaseModel):
    """Free-form metric: any column + statistic, with optional unit conversion.

    The unit fields are excluded from the model's own JSON schema (SkipJsonSchema)
    and re-introduced via a schema `dependencies` block, so the form only shows
    them once "Convert Units" is checked. Units are ignored for count/nunique.
    """

    model_config = ConfigDict(
        title="Custom",
        json_schema_extra={
            "dependencies": {
                "convert_units": {
                    "oneOf": [
                        {"properties": {"convert_units": {"const": False}}},
                        {
                            "properties": {
                                "convert_units": {"const": True},
                                # No `default` on these: a default on a dependency branch
                                # gets seeded into formData even while unchecked, leaving
                                # an orphaned value behind.
                                "original_unit": {
                                    "title": "Original Unit",
                                    "type": "string",
                                    "oneOf": _UNIT_OPTIONS,
                                },
                                "new_unit": {
                                    "title": "New Unit",
                                    "type": "string",
                                    "oneOf": _UNIT_OPTIONS,
                                },
                            }
                        },
                    ]
                }
            }
        },
    )
    metric: Annotated[Literal["custom"], Field(default="custom", title="Metric")] = "custom"
    display_name: Annotated[
        str,
        Field(
            title="Display Name",
            description="Column header shown in the summary table.",
        ),
    ]
    aggregator: Annotated[CustomStatistic, Field(title="Statistic")]
    column: Annotated[str, Field(title="Column", description="Column to aggregate.")]
    convert_units: Annotated[bool, Field(default=False, title="Convert Units")] = False
    original_unit: SkipJsonSchema[Unit | None] = None
    new_unit: SkipJsonSchema[Unit | None] = None
    decimal_places: Annotated[int, Field(default=2, title="Decimal Places")] = 2

    @model_validator(mode="after")
    def check_units(self):
        if self.convert_units and (self.original_unit is None or self.new_unit is None):
            raise ValueError("select both an original and a new unit when unit conversion is enabled")
        return self

    def to_summary_param(self) -> SummaryParam:
        if self.aggregator == "count" or self.aggregator == "nunique":
            return TallySummaryParam(
                display_name=self.display_name,
                aggregator=self.aggregator,
                column=self.column,
            )
        return NumericSummaryParam(
            display_name=self.display_name,
            aggregator=self.aggregator,
            column=self.column,
            original_unit=self.original_unit if self.convert_units else None,
            new_unit=self.new_unit if self.convert_units else None,
            decimal_places=self.decimal_places,
        )


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

_DEFAULT_PATROL_SUMMARY_METRICS: list = [
    {"metric": "patrol_count"},
    {"metric": "total_distance", "unit": "km"},
    {"metric": "total_duration", "unit": "h"},
    {"metric": "patrol_days"},
    {"metric": "area_covered_merged", "swath_width_meters": 500.0},
    {"metric": "area_covered_unmerged", "swath_width_meters": 500.0},
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
