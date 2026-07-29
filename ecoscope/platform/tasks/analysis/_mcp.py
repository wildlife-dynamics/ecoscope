from typing import Annotated, TypeAlias, cast

from pydantic import Field
from pydantic.functional_validators import AfterValidator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._time_density import (
    TimeDensityReturnGDF,
    UDPercentiles,
    _coerce_percentile_strings_to_floats,
)

MCP_DEFAULT_PERCENTILES = ["50", "60", "70", "80", "90", "95", "99.999"]

RelocationsAnnotation: TypeAlias = Annotated[
    AnyGeoDataFrame,
    Field(description="The point relocations geodataframe for a single subject.", exclude=True),
]
McpCrsAnnotation: TypeAlias = Annotated[
    str,
    AdvancedField(
        default="ESRI:102022",
        title="Coordinate Reference System",
        description=(
            "The projected, linear-unit coordinate reference system to rank fixes by distance from"
            " centroid and compute hull areas in - must be a valid CRS authority code, for example ESRI:102022"
        ),
    ),
]
McpPercentileAnnotation: TypeAlias = Annotated[
    list[UDPercentiles] | SkipJsonSchema[list[float]] | SkipJsonSchema[None],
    AdvancedField(
        default=MCP_DEFAULT_PERCENTILES,
        description="Choose the percentile levels (of fixes closest to the centroid) to draw MCP polygons for.",
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
    from ecoscope.analysis.UD import calculate_mcp_range

    if percentiles is not None and len(percentiles) == 0:
        raise ValueError("Percentile values, if provided, cannot be empty.")
    percentiles = (
        sorted(set(percentiles), reverse=True)  # type: ignore[assignment]
        if percentiles is not None
        else [50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.999]
    )

    result = calculate_mcp_range(
        relocations=relocations_gdf,
        percentile_levels=percentiles,  # type: ignore[arg-type]
        crs=crs,
    )
    # subject_id/actual_percentile are dropped here so MCP's task-level output shares
    # TimeDensityReturnGDF's exact schema (percentile, geometry, area_sqkm) with ETD/LTD -
    # both are still available from calculate_mcp_range directly for callers who want them.
    result.drop(columns=["subject_id", "actual_percentile"], inplace=True)
    result["area_sqkm"] = result.area / 1000000.0

    return cast(TimeDensityReturnGDF, result)
