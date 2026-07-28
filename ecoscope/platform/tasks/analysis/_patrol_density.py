from dataclasses import replace
from typing import Annotated, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.tasks.analysis._density_weighting import WeightingSpec, labeled_weighting
from ecoscope.platform.tasks.analysis._time_density import LTD_DEFAULT_PERCENTILES, UDPercentiles
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


class PatrolWeightingSelection(BaseModel):
    model_config = ConfigDict(
        title="",
        json_schema_extra={
            "dependencies": {
                "density_sum_column": {
                    "oneOf": [
                        {
                            "properties": {
                                "density_sum_column": {"enum": ["timespan_seconds", "dist_meters"]},
                            }
                        },
                        {
                            "properties": {
                                "density_sum_column": {"const": "normalised_ltd"},
                                # No `default` here: RJSF resolves dependency
                                # branches when rendering but never copies their
                                # defaults into formData, so a branch default
                                # would show an empty picker. The pre-fill
                                # instead rides on set_patrol_weighting_spec's
                                # param-level object default.
                                "percentiles": {
                                    "title": "Percentile Levels",
                                    "description": "Choose the time density percentile bins to display.",
                                    "type": "array",
                                    "uniqueItems": True,
                                    "items": {"type": "string", "enum": list(get_args(UDPercentiles))},
                                },
                            }
                        },
                    ]
                }
            }
        },
    )
    density_sum_column: Annotated[
        PatrolDensityWeighting,
        Field(
            title="Density Calculation",
            description=(
                "Weight each grid cell by total patrol time or distance travelled,"
                " or by time normalised as a percentage of the total (LTD)."
            ),
            json_schema_extra=labeled_weighting(PATROL_WEIGHTING_SPECS),
        ),
    ] = "timespan_seconds"
    percentiles: SkipJsonSchema[list[float] | None] = None

    @model_validator(mode="after")
    def clear_orphaned_percentiles(self):
        # percentiles left behind by switching away from LTD in the form are ignored
        if self.density_sum_column != "normalised_ltd":
            self.percentiles = None
        return self


@register()
def set_patrol_weighting_spec(
    weighting: Annotated[
        PatrolWeightingSelection | SkipJsonSchema[None],
        Field(
            default=None,
            title="",
            # percentiles seeded here (the patrols Time Density Map defaults)
            # so the picker comes pre-filled when Normalised (LTD) is selected;
            # while another option is selected it sits hidden in formData and
            # clear_orphaned_percentiles discards it server-side.
            json_schema_extra={
                "default": {
                    "density_sum_column": "timespan_seconds",
                    "percentiles": list(LTD_DEFAULT_PERCENTILES),
                }
            },
        ),
    ] = None,
) -> WeightingSpec:
    """
    Select the weighting used for the patrol density grid.

    Percentile levels only apply to the "ltd" weighting (the percentile bins
    to display, as in the patrols workflow's Time Density Map); None keeps the
    LTD defaults. Sum weightings ignore them.
    """
    if weighting is None:
        weighting = PatrolWeightingSelection()
    spec = PATROL_WEIGHTING_SPECS[weighting.density_sum_column]
    if weighting.percentiles:
        spec = replace(spec, percentiles=tuple(weighting.percentiles))
    return spec
