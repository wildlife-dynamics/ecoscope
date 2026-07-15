from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._density_weighting import (
    WeightingSpec,
    density_legend_title,
    labeled_weighting,
    normalize_density,
)
from ecoscope.platform.tasks.transformation._unit import Unit

PatrolDensityWeighting: TypeAlias = Literal["timespan_seconds", "dist_meters"]

# Single source of truth per patrol weighting column; supporting a new
# weighting is one entry here (plus the Literal above).
PATROL_WEIGHTING_SPECS: dict[str, WeightingSpec] = {
    "timespan_seconds": WeightingSpec(Unit.SECOND, Unit.HOUR, "Time", "hours"),
    "dist_meters": WeightingSpec(Unit.METER, Unit.KILOMETER, "Distance", "km"),
}


@register()
def set_density_weighting(
    density_sum_column: Annotated[
        PatrolDensityWeighting,
        Field(
            description="Weight each grid cell by total patrol time or distance travelled.",
            json_schema_extra=labeled_weighting(PATROL_WEIGHTING_SPECS),
        ),
    ] = "timespan_seconds",
) -> PatrolDensityWeighting:
    """
    Select the column used to weight the patrol density grid.
    """
    return density_sum_column


@register()
def normalize_density_units(
    df: Annotated[
        AnyGeoDataFrame,
        Field(description="Feature density output with a raw 'density' column.", exclude=True),
    ],
    density_sum_column: Annotated[
        PatrolDensityWeighting,
        Field(description="The column the density was summed from; determines the display unit."),
    ],
) -> AnyGeoDataFrame:
    """
    Convert a raw density column to display units: seconds to hours, meters to kilometers.
    """
    return normalize_density(df, PATROL_WEIGHTING_SPECS[density_sum_column])


@register()
def get_density_legend_title(
    density_sum_column: Annotated[
        PatrolDensityWeighting,
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
    return density_legend_title(PATROL_WEIGHTING_SPECS[density_sum_column], title_prefix)
