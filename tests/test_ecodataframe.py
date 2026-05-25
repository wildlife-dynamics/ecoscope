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


def test_from_file():
    edf = EcoDataFrame.from_file("tests/sample_data/vector/AOI_sites.gpkg")
    assert isinstance(edf, EcoDataFrame)
    assert len(edf.gdf) > 0


def test_from_features():
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            "properties": {"id": 1},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
            "properties": {"id": 2},
        },
    ]
    edf = EcoDataFrame.from_features(features)
    assert isinstance(edf, EcoDataFrame)
    assert len(edf.gdf) == 2


def test_remove_filtered_warns_on_non_bool_junk_status(sample_gdf):
    sample_gdf["junk_status"] = sample_gdf["junk_status"].astype(int)

    edf = EcoDataFrame(gdf=sample_gdf)
    with pytest.warns(UserWarning, match="junk_status column is of type"):
        edf.remove_filtered(inplace=True)
    assert len(edf.gdf) == 1
