import logging

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from shapely.geometry import MultiPoint  # type: ignore[import-untyped]

from ecoscope import Relocations

logger = logging.getLogger(__name__)


def calculate_mcp_range(
    relocations: Relocations | gpd.GeoDataFrame,
    percentile_levels: list[float],
    crs: str = "EPSG:3857",
    subject_id: str = "",
) -> gpd.GeoDataFrame:
    """Estimate a home range using the Minimum Convex Polygon (MCP) method.

    For each requested percentile, ranks fixes by distance from the centroid
    of all fixes, keeps only the closest `percentile` fraction (dropping the
    most distant fixes as outlier excursions), and returns the convex hull
    enclosing what remains (Mohr 1947's convex hull, refined to exclude
    distant excursions - also the convention used by R's
    `adehabitatHR::mcp()`). Unlike a utilization-distribution method (e.g.
    ETD or BBMM), MCP models no density surface - it is purely geometric and
    produces no raster output.

    Parameters
    ----------
    relocations : Relocations or gpd.GeoDataFrame
        Point relocations to estimate the range from.
    percentile_levels : list[float]
        The percentile levels to compute, e.g. `[50.0, 90.0]`.
    crs : str
        The projected coordinate reference system to rank fixes and compute
        hull areas in - must be a valid CRS authority code, e.g. EPSG:3857.
    subject_id : str
        Value written to the `subject_id` column of the returned rows.

    Returns
    -------
    gpd.GeoDataFrame
        One row per requested percentile that had enough fixes to form a
        polygon (at least 3), with columns `subject_id`, `percentile`,
        `actual_percentile` (the percentile actually achieved, which can
        differ slightly from the requested one due to integer fix counts),
        and `geometry` (the convex hull).
    """
    gdf = relocations.gdf if isinstance(relocations, Relocations) else relocations
    gdf = gdf.to_crs(crs)

    xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    n = len(xy)

    centroid = xy.mean(axis=0)
    distance_from_centroid = np.linalg.norm(xy - centroid, axis=1)
    closest_first = np.argsort(distance_from_centroid, kind="stable")

    rows = []
    for percentile in sorted(percentile_levels, reverse=True):
        fix_count = int(np.floor(percentile / 100.0 * n))
        if fix_count < 3:
            logger.warning(
                f"Skipping {percentile}% MCP: only {fix_count} fix(es) at that level, "
                "need at least 3 to form a polygon."
            )
            continue

        retained = xy[closest_first[:fix_count]]
        rows.append(
            {
                "subject_id": subject_id,
                "percentile": percentile,
                "actual_percentile": 100.0 * fix_count / n,
                "geometry": MultiPoint(retained).convex_hull,
            }
        )

    return gpd.GeoDataFrame(
        rows,
        columns=["subject_id", "percentile", "actual_percentile", "geometry"],
        crs=crs,
    )
