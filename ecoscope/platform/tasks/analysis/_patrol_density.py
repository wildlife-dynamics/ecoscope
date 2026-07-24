from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.tasks.analysis._density_weighting import WeightingSpec, labeled_weighting
from ecoscope.platform.tasks.transformation._unit import Unit

PatrolDensityWeighting: TypeAlias = Literal["timespan_seconds", "dist_meters", "normalised_ltd"]

# Single source of truth per patrol weighting; supporting a new weighting is
# one entry here (plus the Literal above). Keys are the form dropdown values
# and are independent of the sum column ("normalised_ltd" reuses
# timespan_seconds).
PATROL_WEIGHTING_SPECS: dict[str, WeightingSpec] = {
    "timespan_seconds": WeightingSpec("timespan_seconds", Unit.SECOND, Unit.HOUR, "Time"),
    "dist_meters": WeightingSpec("dist_meters", Unit.METER, Unit.KILOMETER, "Distance"),
    "normalised_ltd": WeightingSpec(
        "timespan_seconds",
        Unit.SECOND,
        Unit.PERCENT,
        "Normalised (LTD)",
        mode="ltd",
        legend_label="Time Spent",
    ),
}


@register()
def set_patrol_weighting_spec(
    density_sum_column: Annotated[
        PatrolDensityWeighting,
        Field(
            description=(
                "Weight each grid cell by total patrol time or distance travelled,"
                " or by time normalised as a percentage of the total (LTD)."
            ),
            json_schema_extra=labeled_weighting(PATROL_WEIGHTING_SPECS),
        ),
    ] = "timespan_seconds",
) -> WeightingSpec:
    """
    Select the weighting used for the patrol density grid.
    """
    return PATROL_WEIGHTING_SPECS[density_sum_column]
