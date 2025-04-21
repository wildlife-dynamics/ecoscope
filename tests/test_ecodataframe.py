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


def test_constructor(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
    assert isinstance(edf, EcoDataFrame)
    assert edf._constructor == EcoDataFrame


def test_crs_gdf(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
    assert edf.crs == "epsg:4326"


def test_crs_custom(sample_gdf):
    edf = sample_gdf.copy()
    edf.crs = None
    edf = EcoDataFrame(edf, crs="epsg:3857")
    assert edf.crs == "epsg:3857"


def test_to_crs(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
    edf = edf.to_crs("epsg:3857")
    assert isinstance(edf, EcoDataFrame)
    assert edf.crs == "epsg:3857"


def test_geometry_gdf(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
    assert edf.geometry.name == "my_geometry"


def test_geometry_custom(sample_gdf):
    edf = EcoDataFrame(sample_gdf, geometry="other_geometry")
    assert edf.geometry.name == "other_geometry"


def test_set_geometry(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
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


def test_getitem_series(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
    edf = edf[edf.id > 1]
    assert isinstance(edf, EcoDataFrame)
    assert len(edf) == 1


def test_getitem_list(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
    edf = edf[["id"]]
    assert isinstance(edf, EcoDataFrame)
    assert len(edf) == 2


def test_getitem_slice(sample_gdf):
    edf = EcoDataFrame(sample_gdf)
    edf = edf[slice(1)]
    assert isinstance(edf, EcoDataFrame)
    assert len(edf) == 1


def test_astype(sample_gdf):
    edf = EcoDataFrame({"a": [1], "geometry": [Point(0, 0)]})
    edf = edf.astype("object")
    assert isinstance(edf, EcoDataFrame)


def test_merge(sample_gdf):
    edf1 = EcoDataFrame({"lkey": ["foo", "bar"], "value": [1, 2]})
    edf2 = EcoDataFrame({"rkey": ["foo", "bar"], "value": [5, 6]})
    edf = edf1.merge(edf2, left_on="lkey", right_on="rkey")
    assert isinstance(edf, EcoDataFrame)


def test_dissolve(sample_gdf):
    edf = EcoDataFrame({"a": [1], "geometry": [Point(0, 0)]})
    edf = edf.dissolve(by="a")
    assert isinstance(edf, EcoDataFrame)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_explode(sample_gdf):
    edf = EcoDataFrame({"a": [1], "geometry": [Point(0, 0)]})
    edf = edf.explode()
    assert isinstance(edf, EcoDataFrame)


def test_concat(sample_gdf):
    edf1 = EcoDataFrame({"a": [1], "geometry": [Point(0, 0)]})
    edf2 = EcoDataFrame({"a": [2], "geometry": [Point(1, 1)]})
    edf = pd.concat([edf1, edf2])
    assert isinstance(edf, EcoDataFrame)


def test_plot_series(sample_gdf, mocker):
    series_plot_mock = mocker.patch.object(pd.Series, "plot")
    edf = EcoDataFrame({"x": [1]})
    edf["x"].plot()
    series_plot_mock.assert_called_once_with()


def test_plot_geoseries(sample_gdf, mocker):
    geoseries_plot_mock = mocker.patch.object(gpd.GeoSeries, "plot")
    edf = EcoDataFrame({"geometry": [Point(0, 0)]})
    edf["geometry"].plot()
    geoseries_plot_mock.assert_called_once_with()


def test_plot_dataframe(sample_gdf, mocker):
    dataframe_plot_mock = mocker.patch.object(pd.DataFrame, "plot")
    edf = EcoDataFrame({"x": [1], "y": [1]})
    edf[["x", "y"]].plot()
    dataframe_plot_mock.assert_called_once_with()


def test_plot_geodataframe(sample_gdf, mocker):
    geodataframe_plot_mock = mocker.patch.object(gpd.GeoDataFrame, "plot")
    edf = EcoDataFrame({"x": [1], "geometry": [Point(0, 0)]})
    subset = edf[["x", "geometry"]]
    subset.plot(column="x")
    geodataframe_plot_mock.assert_called_once_with(subset, column="x")
