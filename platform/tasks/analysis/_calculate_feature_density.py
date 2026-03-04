from typing import Annotated, Literal

from ecoscope.platform.annotations import AnyGeoDataFrame
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register


@register()
def calculate_feature_density(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(
            description="The feature data to count or sum per grid cell.", exclude=True
        ),
    ],
    meshgrid: Annotated[
        AnyGeoDataFrame,
        Field(
            description="The grid cells used to aggregate the feature data.",
            exclude=True,
        ),
    ],
    geometry_type: Annotated[
        Literal["point", "line"],
        Field(description="The geometry type of the provided geodataframe"),
    ],
    sum_column: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="Sum values in this column per grid cell, rather than counting rows"
        ),
    ] = None,
) -> AnyGeoDataFrame:
    """
    Count features or sum column values per grid cell.
    """
    from ecoscope.analysis.feature_density import (  # type: ignore[import-untyped]
        calculate_feature_density,
    )

    result = calculate_feature_density(
        selection=geodataframe,
        grid=meshgrid,
        geometry_type=geometry_type,
        sum_column=sum_column,
    )

    return result
