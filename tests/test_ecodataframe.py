import geopandas as gpd
import geopandas.testing
import pytest
from shapely.geometry import Point, box

from ecoscope.base import EcoDataFrame


@pytest.fixture
def sample_gdf():
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "my_geometry": [box(1, 1, 2, 2), box(3, 3, 4, 4)],
            "other_geometry": [Point(0, 0), Point(1, 1)],
            "junk_status": [True, False],
        },
        geometry="my_geometry",
        crs="epsg:4326",
    )


def test_gdf(sample_gdf):
    edf = EcoDataFrame(gdf=sample_gdf)
    assert edf.gdf.crs == "epsg:4326"
    assert edf.gdf.geometry.name == "my_geometry"


def test_reset_filter(sample_gdf):
    edf = EcoDataFrame(gdf=sample_gdf)
    edf.reset_filter(inplace=True)
    assert not edf.gdf["junk_status"].all()


def test_remove_filtered(sample_gdf):
    expected = sample_gdf.drop(index=0)

    edf = EcoDataFrame(gdf=sample_gdf)
    edf.remove_filtered(inplace=True)
    assert len(edf.gdf) == 1
    gpd.testing.assert_geodataframe_equal(edf.gdf, expected)
