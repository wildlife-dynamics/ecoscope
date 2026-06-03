from typing import Annotated, TypeAlias, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame

CrsAnnotation: TypeAlias = Annotated[
    str,
    Field(
        default="EPSG:4326",
        title="Coordinate Reference System",
        description=(
            "The target coordinate reference system, as a valid CRS authority code"
            " (e.g. 'EPSG:4326', 'EPSG:3857', 'ESRI:53042')."
        ),
    ),
]


@register()
def convert_crs(
    df: AnyGeoDataFrame,
    crs: CrsAnnotation = "EPSG:4326",
) -> AnyGeoDataFrame:
    """
    Re-project a GeoDataFrame's geometries to the given CRS.

    Args:
        df: Input GeoDataFrame. Must have CRS metadata set.
        crs: Target CRS authority code (e.g. ``"EPSG:4326"``).

    Returns:
        GeoDataFrame with geometries re-projected to ``crs``.

    Raises:
        ValueError: If the input GeoDataFrame has no CRS metadata.
    """
    if df.crs is None:  # type: ignore[attr-defined]
        raise ValueError(
            "GeoDataFrame has no CRS information. "
            f"Cannot safely convert to {crs} without knowing the source CRS. "
            "Please ensure the source data includes CRS metadata."
        )
    return cast(AnyGeoDataFrame, df.to_crs(crs))  # type: ignore[operator]
