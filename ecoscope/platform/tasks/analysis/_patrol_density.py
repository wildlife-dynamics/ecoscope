from dataclasses import replace
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.tasks.analysis._density_weighting import WeightingSpec, labeled_weighting
from ecoscope.platform.tasks.analysis._time_density import LtdPercentileAnnotation
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


def _reveal_ltd_config_only_for_ltd(schema: dict[str, Any]) -> None:
    """Move LTD-only fields behind an allOf/if/then keyed on the weighting choice.

    Forms then only reveal Percentile Levels when "Normalised (LTD)" is
    selected. The lenient shape is deliberate: no `additionalProperties`, and
    no `default`/`minItems` on the conditional field — RJSF seeds hidden array
    fields from the first render, so a default or minItems would leak values
    or 422 submits in the non-LTD modes; the task falls back to the LTD
    default percentiles when the value is omitted or empty.
    """
    percentiles = schema["properties"].pop("percentiles")
    for key in ("default", "minItems", "ecoscope:advanced"):
        percentiles.pop(key, None)
    schema.pop("additionalProperties", None)
    schema["allOf"] = [
        {
            "if": {"properties": {"density_sum_column": {"const": "normalised_ltd"}}},
            "then": {"properties": {"percentiles": percentiles}},
        }
    ]


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
    percentiles: LtdPercentileAnnotation = None,
) -> WeightingSpec:
    """
    Select the weighting used for the patrol density grid.

    `percentiles` only applies to the "ltd" weighting (the percentile bins to
    display, as in the patrols workflow's Time Density Map); None keeps the
    LTD defaults. Sum weightings ignore it.
    """
    spec = PATROL_WEIGHTING_SPECS[density_sum_column]
    if percentiles:
        spec = replace(spec, percentiles=tuple(float(p) for p in percentiles))
    return spec


# Task-level schema hook (wt-registry >= the version adding
# @register(json_schema_extra=...)). Set as an attribute rather than a
# decorator kwarg so older wt-registry versions simply ignore it (the form
# then shows percentiles unconditionally) instead of failing at import.
set_patrol_weighting_spec.__wt_json_schema_extra__ = (  # type: ignore[attr-defined]
    _reveal_ltd_config_only_for_ltd
)
