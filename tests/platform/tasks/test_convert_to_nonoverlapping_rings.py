import geopandas as gpd  # type: ignore[import-untyped]
from shapely.geometry import Polygon

from ecoscope.platform.tasks.transformation import (
    apply_rings_correction_if_enabled,
    convert_to_nonoverlapping_rings,
)


def _nested_gdf() -> gpd.GeoDataFrame:
    """Three concentric squares - 10x10, 20x20, 30x30 - centered on the origin,
    tagged with ascending percentile levels. Nested/cumulative, like a real
    percentile home-range polygon set (each larger percentile contains the
    smaller ones)."""
    inner = Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5)])
    middle = Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)])
    outer = Polygon([(-15, -15), (-15, 15), (15, 15), (15, -15)])
    return gpd.GeoDataFrame(
        {"percentile": [50.0, 80.0, 99.0], "area_sqkm": [100.0, 400.0, 900.0]},
        geometry=[inner, middle, outer],
        crs="EPSG:3857",
    )


def test_convert_to_nonoverlapping_rings_removes_overlap():
    result = convert_to_nonoverlapping_rings(_nested_gdf())

    # No two rings should overlap (only touch, if anything, at shared boundaries).
    geoms = result.geometry.tolist()
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            assert geoms[i].intersection(geoms[j]).area == 0


def test_convert_to_nonoverlapping_rings_areas_sum_to_outermost():
    result = convert_to_nonoverlapping_rings(_nested_gdf())
    # 10x10 + (20x20 - 10x10) + (30x30 - 20x20) == 30x30
    assert sum(result.geometry.area) == 30 * 30


def test_convert_to_nonoverlapping_rings_sorted_descending_by_percentile():
    # Feed it out of order to confirm the sort is the function's own responsibility.
    shuffled = _nested_gdf().iloc[[1, 2, 0]].reset_index(drop=True)
    result = convert_to_nonoverlapping_rings(shuffled)
    assert list(result["percentile"]) == [99.0, 80.0, 50.0]


def test_convert_to_nonoverlapping_rings_preserves_other_columns():
    result = convert_to_nonoverlapping_rings(_nested_gdf())
    # area_sqkm should stay as originally computed (cumulative area), not be recalculated
    # from the new ring geometry - only `geometry` itself is replaced.
    assert list(result["area_sqkm"]) == [900.0, 400.0, 100.0]


def test_convert_to_nonoverlapping_rings_single_row_is_unchanged():
    gdf = _nested_gdf().iloc[[0]].reset_index(drop=True)
    result = convert_to_nonoverlapping_rings(gdf)
    assert result.geometry.iloc[0].equals(gdf.geometry.iloc[0])


def test_apply_rings_correction_if_enabled_true_matches_convert_to_nonoverlapping_rings():
    gdf = _nested_gdf()
    enabled = apply_rings_correction_if_enabled(gdf, enabled=True)
    direct = convert_to_nonoverlapping_rings(gdf)
    assert list(enabled["percentile"]) == list(direct["percentile"])
    assert all(a.equals(b) for a, b in zip(enabled.geometry, direct.geometry))


def test_apply_rings_correction_if_enabled_false_passes_through_unchanged():
    gdf = _nested_gdf()
    result = apply_rings_correction_if_enabled(gdf, enabled=False)
    assert list(result["percentile"]) == list(gdf["percentile"])
    assert all(a.equals(b) for a, b in zip(result.geometry, gdf.geometry))
