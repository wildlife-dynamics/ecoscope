import logging

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from shapely.geometry import MultiPoint  # type: ignore[import-untyped]

from ecoscope.relocations import Relocations

logger = logging.getLogger(__name__)


def calculate_mcp_range(
    relocations: Relocations | gpd.GeoDataFrame,
    percentile_levels: list[float],
    crs: str = "ESRI:102022",
    subject_id: str = "",
) -> gpd.GeoDataFrame:
    """
    The Minimum Convex Polygon (MCP) is the oldest and simplest home-range estimator: for a given
    percentile, it ranks relocations by their distance from the centroid of all relocations, keeps
    only the closest `percentile` fraction (dropping the most distant fixes as outlier excursions),
    and draws the convex hull enclosing what remains. This is the classic percentile-MCP convention
    (Mohr 1947's original 100% convex hull, refined to exclude distant excursions) also used by R's
    `adehabitatHR::mcp()`.

    Unlike a utilization-distribution method (e.g. `calculate_etd_range`), MCP models no density
    surface at all - it is purely geometric, and consequently produces no raster output. It is also
    the most heavily criticized home-range estimator in current use: because it only cares about the
    outer boundary of a fix subset, it commonly includes large areas the animal never actually visited.

    Parameters
    ----------
    relocations : Relocations or GeoDataFrame
        Point relocations for a single subject.
    percentile_levels : list of float
        Percentile levels in the range (0, 100] to compute, e.g. [50, 80, 99].
    crs : str, default "ESRI:102022"
        A projected, linear-unit coordinate reference system to compute distances and areas in - `relocations`
        is reprojected into this CRS before any distance calculation, since MCP's ranking-by-distance step is
        only meaningful in a projected space. Defaults to the Africa Albers Equal Area Conic projection.
    subject_id : str, optional
        Identifier stamped onto every output row.

    Returns
    -------
    GeoDataFrame
        One row per percentile level with at least 3 retained fixes (fewer are skipped, with a
        warning, since a polygon needs at least 3 points), with columns:

        - `subject_id`: as passed in.
        - `percentile`: the requested level.
        - `actual_percentile`: the level actually achieved. Retaining a whole number of fixes means
          rounding down from the exact fractional target, so this is usually a little below `percentile`.
        - `geometry`: the hull polygon, in `crs`.

        Rows are ordered by descending `percentile`, matching `ecoscope.analysis.percentile.get_percentile_area`.
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

    return gpd.GeoDataFrame(rows, columns=["subject_id", "percentile", "actual_percentile", "geometry"], crs=crs)
