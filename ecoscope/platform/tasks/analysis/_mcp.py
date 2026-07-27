from typing import Annotated, Any, TypeAlias, cast

import pandera.pandas as pa
import pandera.typing as pa_typing
from pydantic import Field
from pydantic.functional_validators import AfterValidator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    AnyGeoDataFrame,
    DataFrame,
    JsonSerializableDataFrameModel,
)
from ecoscope.platform.tasks.analysis._time_density import UDPercentiles, _coerce_percentile_strings_to_floats


class McpReturnGDFSchema(JsonSerializableDataFrameModel):
    percentile: pa_typing.Series[float] = pa.Field()
    actual_percentile: pa_typing.Series[float] = pa.Field()
    geometry: pa_typing.Series[Any] = pa.Field()  # see note in annotations.py
    area_sqkm: pa_typing.Series[float] = pa.Field()


McpReturnGDF: TypeAlias = DataFrame[McpReturnGDFSchema]

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
) -> McpReturnGDF:
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
    result.drop(columns="subject_id", inplace=True)
    result["area_sqkm"] = result.area / 1000000.0

    return cast(McpReturnGDF, result)
