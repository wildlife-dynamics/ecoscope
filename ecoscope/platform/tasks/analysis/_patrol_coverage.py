from typing import Annotated, Literal

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame


def _coverage_area_km2(
    gdf: AnyGeoDataFrame,
    swath_width_meters: float,
    merged: bool,
    area_crs: str,
) -> float:
    """
    Compute the ground area (km²) covered by buffering track segments.

    Each segment is buffered by ``swath_width_meters / 2`` per side to form a
    corridor of full width ``swath_width_meters``. When ``merged`` is True the
    area of the union of all corridors is returned (overlaps counted once);
    otherwise the per-segment corridor areas are summed.
    """
    if gdf.empty:
        return 0.0

    buffers = gdf.to_crs(area_crs).geometry.buffer(swath_width_meters / 2)

    if merged:
        area_m2 = buffers.union_all().area
    else:
        area_m2 = buffers.area.sum()

    return area_m2 / 1e6


@register()
def calculate_patrol_coverage(
    trajectory_gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="LineString trajectory segments to measure coverage for.", exclude=True),
    ],
    swath_width_meters: Annotated[
        float,
        Field(
            description="Full corridor width in meters; each segment is buffered by half this per side.",
        ),
    ] = 500.0,
    mode: Annotated[
        Literal["merged", "unmerged"],
        Field(
            description=(
                "merged = area of the union of buffered segments (overlaps counted once); "
                "unmerged = sum of per-segment buffered areas."
            ),
        ),
    ] = "merged",
    area_crs: Annotated[
        str,
        AdvancedField(
            default="EPSG:6933",
            description="Equal-area CRS used to measure area.",
        ),
    ] = "EPSG:6933",
) -> Annotated[float, Field(description="Area covered in km²")]:
    """
    Return the ground area (km²) covered by patrol track segments.

    Args:
        trajectory_gdf: LineString trajectory segments (must have geometry).
        swath_width_meters: Full corridor width; each segment is buffered by
            swath_width_meters / 2 per side.
        mode: "merged" measures the union of buffered segments; "unmerged" sums
            the per-segment buffered areas.
        area_crs: Equal-area CRS used to measure area.

    Returns:
        The covered area in square kilometers.
    """
    return _coverage_area_km2(
        trajectory_gdf,
        swath_width_meters,
        merged=(mode == "merged"),
        area_crs=area_crs,
    )
