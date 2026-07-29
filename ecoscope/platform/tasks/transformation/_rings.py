from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def convert_to_nonoverlapping_rings(percentiles_gdf: AnyGeoDataFrame) -> AnyGeoDataFrame:
    """
    Convert nested/cumulative percentile polygons into non-overlapping annular rings.

    Percentile home-range polygons are typically cumulative: each larger percentile's
    polygon fully contains all smaller ones' (a convex hull is monotonic under point-set
    inclusion, and a raster iso-area's super-level-set only grows as the percentile
    threshold rises). Rendered as stacked semi-transparent map layers, that nesting means
    a location covered by N overlapping polygons shows at compounded opacity
    (1 - (1 - opacity)^N) rather than the opacity actually requested - most visible at
    low opacity, where inner percentile bands can look nearly solid.

    This task removes the overlap: each output ring is that percentile's polygon minus
    the next-smaller percentile's polygon, so no two rings overlap and a single opacity
    value renders uniformly everywhere. It only replaces the `geometry` column - other
    properties (e.g. an already-computed `area_sqkm`, which should still reflect the
    original cumulative area) are left untouched.

    Args:
        percentiles_gdf: A GeoDataFrame with a `percentile` column and a `geometry`
            column of cumulative/nested polygons, one row per percentile level.

    Returns:
        The same GeoDataFrame, sorted descending by `percentile`, with `geometry`
        replaced by non-overlapping rings.
    """
    ascending = percentiles_gdf.sort_values("percentile").reset_index(drop=True).copy()  # type: ignore[attr-defined]
    prev_geom = None
    rings = []
    for geom in ascending.geometry:
        rings.append(geom.difference(prev_geom) if prev_geom is not None else geom)
        prev_geom = geom
    ascending["geometry"] = rings
    return ascending.iloc[::-1].reset_index(drop=True)  # type: ignore[return-value]


@register()
def apply_rings_correction_if_enabled(percentiles_gdf: AnyGeoDataFrame, enabled: bool) -> AnyGeoDataFrame:
    """A boolean-gated wrapper around convert_to_nonoverlapping_rings, so a pipeline can
    expose it as a user-facing on/off toggle (e.g. a checkbox in a settings form) rather
    than always applying it - spec.yaml itself has no conditional-transform construct,
    only conditional task-skipping, which isn't the same thing (a skipped task cascades
    a skip to its dependents, it doesn't pass its input through unchanged)."""
    return convert_to_nonoverlapping_rings(percentiles_gdf) if enabled else percentiles_gdf
