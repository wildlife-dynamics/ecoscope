from typing import Annotated, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def filter_by_geometry_type(
    df: AnyGeoDataFrame,
    geometry_types: Annotated[
        list[str],
        Field(
            description=("Shapely geometry type names to keep " "(e.g. ['Point'], ['Polygon', 'MultiPolygon'])."),
        ),
    ],
) -> AnyGeoDataFrame:
    """
    Filter a GeoDataFrame to rows whose geometry type is in ``geometry_types``.

    Args:
        df: Input GeoDataFrame.
        geometry_types: List of shapely ``geom_type`` names to retain.

    Returns:
        GeoDataFrame containing only rows whose geometry's ``geom_type`` is in
        ``geometry_types``.
    """
    filtered = df[df.geometry.geom_type.isin(geometry_types)]
    return cast(AnyGeoDataFrame, filtered)
