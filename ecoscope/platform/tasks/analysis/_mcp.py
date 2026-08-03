from typing import Annotated, TypeAlias, cast

from pydantic import Field
from pydantic.functional_validators import AfterValidator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._time_density import (
    DEFAULT_PERCENTILES,
    TimeDensityReturnGDF,
    UDPercentiles,
    _coerce_percentile_strings_to_floats,
)

RelocationsAnnotation: TypeAlias = Annotated[
    AnyGeoDataFrame,
    Field(
        description="The point relocations geodataframe for a single subject.",
        exclude=True,
    ),
]
McpCrsAnnotation: TypeAlias = Annotated[
    str,
    AdvancedField(
        default="ESRI:102022",
        title="Coordinate Reference System",
        description=(
            "The projected coordinate reference system used to rank fixes by distance from"
            " the centroid and compute hull areas - must be a valid CRS code, e.g. ESRI:102022."
        ),
    ),
]
McpPercentileAnnotation: TypeAlias = Annotated[
    list[UDPercentiles] | SkipJsonSchema[list[float]] | SkipJsonSchema[None],
    AdvancedField(
        default=DEFAULT_PERCENTILES,
        description="Choose the percentile levels to display.",
        title="Percentile Levels",
        json_schema_extra={"uniqueItems": True, "minItems": 1},
    ),
    AfterValidator(_coerce_percentile_strings_to_floats),
]


@register()
def calculate_minimum_convex_polygon(
    relocations_gdf: RelocationsAnnotation,
    crs: McpCrsAnnotation = "ESRI:102022",
    percentiles: McpPercentileAnnotation = None,
) -> TimeDensityReturnGDF:
    """Estimate a Minimum Convex Polygon (MCP) home range and return it in the
    same percentile/geometry/area_sqkm shape as the time-density methods, so
    downstream map/table steps don't need to know which method produced it."""
    from ecoscope.analysis.UD import calculate_mcp_range

    if percentiles is not None and len(percentiles) == 0:
        raise ValueError("Percentile values, if provided, cannot be empty.")
    percentiles = (
        sorted(set(percentiles), reverse=True)  # type: ignore[assignment]
        if percentiles is not None
        else [99.999, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0]
    )

    result = calculate_mcp_range(
        relocations=relocations_gdf,
        percentile_levels=percentiles,  # type: ignore[arg-type]
        crs=crs,
    )
    result = result.drop(columns=["subject_id", "actual_percentile"])
    result["area_sqkm"] = result.area / 1_000_000.0

    return cast(TimeDensityReturnGDF, result)
