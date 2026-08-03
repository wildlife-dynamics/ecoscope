import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Point

from ecoscope.platform.tasks.transformation._rings import (
    apply_rings_correction_if_enabled,
    convert_to_nonoverlapping_rings,
)


def _nested_percentiles_gdf():
    # Two concentric circles - the 90th percentile polygon fully contains
    # the 50th percentile one, matching a real cumulative percentile output.
    return gpd.GeoDataFrame(
        {
            "percentile": [50.0, 90.0],
            "geometry": [
                Point(0, 0).buffer(1.0),
                Point(0, 0).buffer(2.0),
            ],
        },
        crs="EPSG:3857",
    )


def test_convert_to_nonoverlapping_rings_removes_overlap():
    result = convert_to_nonoverlapping_rings(_nested_percentiles_gdf())

    # Largest percentile first, matching draw order (paint big ring, then small on top).
    assert list(result["percentile"]) == [90.0, 50.0]

    ring_90 = result.loc[result["percentile"] == 90.0, "geometry"].iloc[0]
    ring_50 = result.loc[result["percentile"] == 50.0, "geometry"].iloc[0]

    # The 90th-percentile ring no longer contains the 50th-percentile polygon.
    assert not ring_90.intersects(ring_50.buffer(-0.01))
    # The 50th-percentile "ring" is unchanged - it has no smaller percentile to subtract.
    assert ring_50.equals(Point(0, 0).buffer(1.0))


def test_convert_to_nonoverlapping_rings_area_sums_to_original_largest():
    original = _nested_percentiles_gdf()
    result = convert_to_nonoverlapping_rings(original)

    original_largest_area = original.geometry.iloc[1].area
    rings_total_area = result.geometry.area.sum()

    assert rings_total_area == pytest.approx(original_largest_area)


def test_apply_rings_correction_if_enabled_true_converts():
    original = _nested_percentiles_gdf()
    result = apply_rings_correction_if_enabled(original, enabled=True)

    ring_90 = result.loc[result["percentile"] == 90.0, "geometry"].iloc[0]
    assert ring_90.area < original.loc[original["percentile"] == 90.0, "geometry"].iloc[0].area


def test_apply_rings_correction_if_enabled_false_passes_through():
    original = _nested_percentiles_gdf()
    result = apply_rings_correction_if_enabled(original, enabled=False)

    assert result is original
