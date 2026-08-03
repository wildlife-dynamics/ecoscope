from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame


def convert_to_nonoverlapping_rings(percentiles_gdf: AnyGeoDataFrame) -> AnyGeoDataFrame:
    """Convert cumulative, nested percentile polygons into non-overlapping rings.

    Percentile home-range polygons are typically cumulative - each larger
    percentile's polygon fully contains all smaller ones' - so rendering them
    as stacked, semi-transparent map layers compounds opacity at the
    overlaps rather than showing the opacity actually requested (most
    visible at low opacity). Subtracting each polygon from the next-larger
    one turns them into a set of non-overlapping rings instead, so opacity
    renders uniformly across the map.

    Args:
        percentiles_gdf: One row per percentile, with a `percentile` column
            and cumulative/nested polygon geometries.

    Returns:
        The same rows, with `geometry` replaced by non-overlapping rings,
        ordered from largest to smallest percentile.
    """
    ascending = percentiles_gdf.sort_values("percentile").reset_index(drop=True).copy()
    prev_geom = None
    rings = []
    for geom in ascending.geometry:
        rings.append(geom.difference(prev_geom) if prev_geom is not None else geom)
        prev_geom = geom
    ascending["geometry"] = rings
    return ascending.iloc[::-1].reset_index(drop=True)


@register()
def apply_rings_correction_if_enabled(percentiles_gdf: AnyGeoDataFrame, enabled: bool) -> AnyGeoDataFrame:
    """Apply `convert_to_nonoverlapping_rings` only if `enabled`, otherwise
    pass `percentiles_gdf` through unchanged. A user-facing on/off toggle for
    the ring conversion, since not every consumer wants cumulative polygons
    replaced with rings.
    """
    return convert_to_nonoverlapping_rings(percentiles_gdf) if enabled else percentiles_gdf
