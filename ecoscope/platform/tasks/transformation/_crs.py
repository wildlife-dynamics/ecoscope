import logging
from typing import Annotated, TypeAlias, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame

logger = logging.getLogger(__name__)

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


@register()
def ensure_wgs84(
    df: AnyGeoDataFrame,
) -> AnyGeoDataFrame:
    """
    Reproject a GeoDataFrame to WGS84 (EPSG:4326), assuming WGS84 (with a
    warning) if it has no CRS set, rather than raising like `convert_crs`
    does - useful for a task whose caller may not control/guarantee the
    input's CRS metadata (e.g. a user-uploaded area of interest).

    Args:
        df: Input GeoDataFrame.

    Returns:
        GeoDataFrame with geometries in EPSG:4326.
    """
    if df.crs is None:  # type: ignore[attr-defined]
        logger.warning("GeoDataFrame has no CRS set; assuming WGS84.")
        return cast(AnyGeoDataFrame, df.set_crs(4326))  # type: ignore[operator]
    if df.crs.to_epsg() != 4326:  # type: ignore[attr-defined]
        return cast(AnyGeoDataFrame, df.to_crs(4326))  # type: ignore[operator]
    return df
