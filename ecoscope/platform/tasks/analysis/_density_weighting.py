from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit

DensityWeighting: TypeAlias = Literal["timespan_seconds", "dist_meters"]


@dataclass(frozen=True)
class _WeightingSpec:
    original_unit: Unit  # unit of the raw summed column
    display_unit: Unit  # unit shown on the map
    option_label: str  # form dropdown label
    unit_word: str  # spelled-out unit for the legend title


# Single source of truth per weighting column; supporting a new weighting is
# one entry here (plus the Literal above).
WEIGHTING_SPECS: dict[str, _WeightingSpec] = {
    "timespan_seconds": _WeightingSpec(Unit.SECOND, Unit.HOUR, "Time", "hours"),
    "dist_meters": _WeightingSpec(Unit.METER, Unit.KILOMETER, "Distance", "km"),
}


def _labeled_weighting(schema: dict) -> None:
    """Field-level json_schema_extra: swap the Literal's bare enum for labeled options."""
    schema.pop("enum", None)
    schema["oneOf"] = [{"const": v, "title": spec.option_label} for v, spec in WEIGHTING_SPECS.items()]


@register()
def set_density_weighting(
    sum_column: Annotated[
        DensityWeighting,
        Field(
            description="Weight each grid cell by total patrol time or distance travelled.",
            json_schema_extra=_labeled_weighting,
        ),
    ] = "timespan_seconds",
) -> DensityWeighting:
    """
    Select the column used to weight the patrol density grid.
    """
    return sum_column


@register()
def normalize_density_units(
    df: Annotated[
        AnyGeoDataFrame,
        Field(description="Feature density output with a raw 'density' column.", exclude=True),
    ],
    sum_column: Annotated[
        DensityWeighting,
        Field(description="The column the density was summed from; determines the display unit."),
    ],
) -> AnyGeoDataFrame:
    """
    Convert a raw density column to display units: seconds to hours, meters to kilometers.
    """
    spec = WEIGHTING_SPECS[sum_column]
    df["density"] = df["density"] * with_unit(1.0, spec.original_unit, spec.display_unit).value
    return df


@register()
def get_density_legend_title(
    sum_column: Annotated[
        DensityWeighting,
        Field(description="The column the density was summed from; determines the display unit."),
    ],
    title_prefix: Annotated[
        str,
        Field(description="Legend title text preceding the display unit."),
    ] = "Patrol Effort",
) -> str:
    """
    Legend title for the density map, including the display unit.
    """
    return f"{title_prefix} ({WEIGHTING_SPECS[sum_column].unit_word})"
