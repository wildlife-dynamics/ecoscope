from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.tasks.analysis._density_weighting import WeightingSpec, labeled_weighting
from ecoscope.platform.tasks.transformation._unit import Unit

PatrolDensityWeighting: TypeAlias = Literal["timespan_seconds", "dist_meters"]

# Single source of truth per patrol weighting column; supporting a new
# weighting is one entry here (plus the Literal above).
PATROL_WEIGHTING_SPECS: dict[str, WeightingSpec] = {
    spec.density_sum_column: spec
    for spec in (
        WeightingSpec("timespan_seconds", Unit.SECOND, Unit.HOUR, "Time"),
        WeightingSpec("dist_meters", Unit.METER, Unit.KILOMETER, "Distance"),
    )
}


@register()
def set_patrol_weighting_spec(
    density_sum_column: Annotated[
        PatrolDensityWeighting,
        Field(
            description="Weight each grid cell by total patrol time or distance travelled.",
            json_schema_extra=labeled_weighting(PATROL_WEIGHTING_SPECS),
        ),
    ] = "timespan_seconds",
) -> WeightingSpec:
    """
    Select the weighting used for the patrol density grid.
    """
    return PATROL_WEIGHTING_SPECS[density_sum_column]
