import geopandas as gpd
import pandas as pd
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
        },
        geometry="my_geometry",
        crs="epsg:4326",
    )


def test_crs_gdf(sample_gdf):
    edf = EcoDataFrame(gdf=sample_gdf)
    assert edf.crs == "epsg:4326"


def test_to_crs(sample_gdf):
    edf = EcoDataFrame(gdf=sample_gdf)
    edf = edf.to_crs("epsg:3857")
    assert isinstance(edf, EcoDataFrame)
    assert edf.crs == "epsg:3857"


def test_geometry_gdf(sample_gdf):
    edf = EcoDataFrame(gdf=sample_gdf)
    assert edf.geometry.name == "my_geometry"


def test_set_geometry(sample_gdf):
    edf = EcoDataFrame(gdf=sample_gdf)
    edf = edf.set_geometry("other_geometry")
    assert isinstance(edf, EcoDataFrame)
    assert edf.geometry.name == "other_geometry"


def test_from_file(sample_gdf, mocker):
    mock = mocker.patch.object(gpd.GeoDataFrame, "from_file")
    edf = EcoDataFrame.from_file("file_name.geojson")
    mock.assert_called_once_with("file_name.geojson")
    assert isinstance(edf, EcoDataFrame)


def test_from_features(sample_gdf, mocker):
    mock = mocker.patch.object(gpd.GeoDataFrame, "from_features")
    edf = EcoDataFrame.from_features("__features__")
    mock.assert_called_once_with("__features__")
    assert isinstance(edf, EcoDataFrame)


def test_dissolve():
    edf = EcoDataFrame(gdf=gpd.GeoDataFrame({"a": [1], "geometry": [Point(0, 0)]}))
    edf = edf.dissolve(by="a")
    assert isinstance(edf, EcoDataFrame)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_explode():
    edf = EcoDataFrame(gdf=gpd.GeoDataFrame({"a": [1], "geometry": [Point(0, 0)]}))
    edf = edf.explode()
    assert isinstance(edf, EcoDataFrame)


def test_concat():
    edf1 = EcoDataFrame(gdf=gpd.GeoDataFrame({"a": [1], "geometry": [Point(0, 0)]}))
    edf2 = EcoDataFrame(gdf=gpd.GeoDataFrame({"a": [2], "geometry": [Point(1, 1)]}))
    edf = pd.concat([edf1, edf2])
    assert isinstance(edf, EcoDataFrame)
