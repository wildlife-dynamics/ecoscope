import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.analysis.UD import calculate_mcp_range
from ecoscope.relocations import Relocations


@pytest.fixture
def clustered_relocs_gdf():
    """A tight 21x21 grid of fixes (1m spacing, centered on the origin) plus two fixes
    ~10km away. The grid points are all far closer to the centroid than the two outliers,
    so percentile cutoffs below 100% are guaranteed to exclude exactly the outliers."""
    grid_x, grid_y = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    core_points = [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    outliers = [Point(10_000, 10_000), Point(-10_000, -10_000)]
    return gpd.GeoDataFrame(geometry=core_points + outliers, crs="ESRI:102022")


@pytest.fixture
def synthetic_relocs(clustered_relocs_gdf):
    gdf = clustered_relocs_gdf.copy()
    gdf["groupby_col"] = "subject-1"
    gdf["fixtime"] = pd.date_range("2020-01-01", periods=len(gdf), freq="1h", tz="UTC")
    return Relocations.from_gdf(gdf, groupby_col="groupby_col", time_col="fixtime")


def test_calculate_mcp_range_accepts_geodataframe(clustered_relocs_gdf):
    result = calculate_mcp_range(clustered_relocs_gdf, percentile_levels=[100.0], crs="ESRI:102022")
    assert len(result) == 1
    assert result.iloc[0].geometry.covers(Point(10_000, 10_000))


def test_calculate_mcp_range_accepts_relocations(synthetic_relocs):
    result = calculate_mcp_range(synthetic_relocs, percentile_levels=[100.0], crs="ESRI:102022")
    assert len(result) == 1
    assert result.iloc[0].geometry.covers(Point(10_000, 10_000))


def test_calculate_mcp_range_excludes_outliers_below_100_percent(clustered_relocs_gdf):
    n = len(clustered_relocs_gdf)  # 441 grid points + 2 outliers = 443
    result = calculate_mcp_range(clustered_relocs_gdf, percentile_levels=[99.0], crs="ESRI:102022", subject_id="s1")

    row = result.iloc[0]
    assert row["subject_id"] == "s1"
    assert row["percentile"] == 99.0
    # floor(0.99 * 443) = 438, so both distant outliers (the 2 farthest fixes) are dropped
    assert row["actual_percentile"] == pytest.approx(100.0 * 438 / n)
    assert not row.geometry.contains(Point(10_000, 10_000))
    assert not row.geometry.contains(Point(-10_000, -10_000))


def test_calculate_mcp_range_area_grows_with_percentile(clustered_relocs_gdf):
    result = calculate_mcp_range(clustered_relocs_gdf, percentile_levels=[50.0, 90.0, 100.0], crs="ESRI:102022")
    areas = result.set_index("percentile").sort_index().geometry.area
    assert areas.loc[50.0] < areas.loc[90.0] < areas.loc[100.0]


def test_calculate_mcp_range_rows_sorted_descending_by_percentile(clustered_relocs_gdf):
    result = calculate_mcp_range(clustered_relocs_gdf, percentile_levels=[50.0, 99.0, 75.0], crs="ESRI:102022")
    assert list(result["percentile"]) == [99.0, 75.0, 50.0]


def test_calculate_mcp_range_skips_percentiles_below_three_points(caplog):
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], crs="ESRI:102022")
    with caplog.at_level(logging.WARNING):
        result = calculate_mcp_range(gdf, percentile_levels=[100.0], crs="ESRI:102022")
    assert result.empty
    assert any("need at least 3" in record.message for record in caplog.records)


def test_calculate_mcp_range_reprojects_to_requested_crs(clustered_relocs_gdf):
    geographic_gdf = clustered_relocs_gdf.to_crs(4326)
    result = calculate_mcp_range(geographic_gdf, percentile_levels=[100.0], crs="ESRI:102022")
    assert result.crs.to_string() == "ESRI:102022"
