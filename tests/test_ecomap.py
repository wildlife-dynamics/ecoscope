import ee
import geopandas as gpd
import pandas as pd
import pytest
import ecoscope
import shapely
from ecoscope.mapping import EcoMap
from ecoscope.analysis.geospatial import datashade_gdf
from lonboard._layer import BitmapLayer, BitmapTileLayer, PathLayer, PolygonLayer, ScatterplotLayer
from lonboard._deck_widget import (
    NorthArrowWidget,
    ScaleWidget,
    TitleWidget,
    SaveImageWidget,
    FullscreenWidget,
)


@pytest.fixture
def poly_gdf():
    gdf = gpd.GeoDataFrame.from_file("tests/sample_data/vector/maec_4zones_UTM36S.gpkg")
    return gdf


@pytest.fixture
def line_gdf():
    gdf = pd.read_csv("/home/alex/Code/ecoscope/tests/sample_data/vector/KDB025Z.csv", index_col="id")
    gdf["geometry"] = gdf["geometry"].apply(lambda x: shapely.wkt.loads(x))
    gdf = ecoscope.base.Relocations.from_gdf(gpd.GeoDataFrame(gdf, crs=4326))
    gdf = ecoscope.base.Trajectory.from_relocations(gdf)
    return gdf


@pytest.fixture
def point_gdf():
    gdf = gpd.GeoDataFrame.from_file("tests/sample_data/vector/observations.geojson")
    return gdf


def test_ecomap_base():
    m = EcoMap()

    assert len(m.deck_widgets) == 3
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)
    assert isinstance(m.deck_widgets[0], FullscreenWidget)
    assert isinstance(m.deck_widgets[1], ScaleWidget)
    assert isinstance(m.deck_widgets[2], SaveImageWidget)


def test_static_map():
    m = EcoMap(static=True)

    assert m.controller is False
    assert len(m.deck_widgets) == 1
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)
    assert isinstance(m.deck_widgets[0], ScaleWidget)


def test_add_legend():
    m = EcoMap(default_widgets=False)
    m.add_legend(labels=["Black", "White"], colors=["#000000", "#FFFFFF"])
    assert len(m.deck_widgets) == 1


def test_add_north_arrow():
    m = EcoMap()
    m.add_north_arrow()
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], NorthArrowWidget)


def test_add_scale_bar():
    m = EcoMap()
    m.add_scale_bar()
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], ScaleWidget)


def test_add_title():
    m = EcoMap()
    m.add_title("THIS IS A TEST TITLE")
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], TitleWidget)


def test_add_save_image():
    m = EcoMap()
    m.add_save_image()
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], SaveImageWidget)


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_image():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.Image("USGS/SRTMGL1_003")
    m.add_ee_layer(ee_object, vis_params)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapTileLayer)


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_image_collection():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5}
    ee_object = ee.ImageCollection("MODIS/006/MCD43C3")
    m.add_ee_layer(ee_object, vis_params)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapTileLayer)


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_feature_collection():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002/GEDI02_A_2021244154857_O15413_04_T05622_02_003_02_V002")
    m.add_ee_layer(ee_object, vis_params)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapTileLayer)


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_geometry():
    m = EcoMap()
    rectangle = ee.Geometry.Rectangle([-40, -20, 40, 20])
    m.add_ee_layer(rectangle, None)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], PolygonLayer)


def test_add_polyline(line_gdf):
    m = EcoMap()
    m.add_layer(m.polyline_layer(line_gdf, get_color=[100, 200, 100], get_width=200))
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], PathLayer)
    assert m.layers[1].get_color == [100, 200, 100, 255]


def test_add_point(point_gdf):
    m = EcoMap()
    m.add_layer(m.point_layer(point_gdf, get_fill_color=[25, 100, 25, 100]))
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], ScatterplotLayer)
    # default color
    assert m.layers[1].get_fill_color == [25, 100, 25, 100]


def test_add_polygon(poly_gdf):
    m = EcoMap()
    m.add_layer(m.polygon_layer(poly_gdf, extruded=True, get_line_width=35), zoom=True)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], PolygonLayer)
    assert m.layers[1].extruded
    assert m.layers[1].get_line_width == 35
    # validating zoom param by checking view state is non-default
    assert m.view_state.longitude != 10
    assert m.view_state.latitude != 0


def test_zoom_to_gdf():
    m = EcoMap()
    x1 = 34.683838
    y1 = -3.173425
    x2 = 38.869629
    y2 = 0.109863
    gs = gpd.GeoSeries.from_wkt(
        [f"POINT ({x1} {y1})", f"POINT ({x2} {y1})", f"POINT ({x1} {y2})", f"POINT ({x2} {y2})"]
    )
    gs = gs.set_crs("EPSG:4326")
    m.zoom_to_bounds(feat=gpd.GeoDataFrame(geometry=gs))

    assert m.view_state.longitude == (x1 + x2) / 2
    assert m.view_state.latitude == (y1 + y2) / 2


def test_add_geotiff():
    m = EcoMap()
    m.add_geotiff("tests/sample_data/raster/uint8.tif", cmap=None)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)


def test_add_geotiff_with_cmap():
    m = EcoMap()
    m.add_geotiff("tests/sample_data/raster/uint8.tif", cmap="jet")
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)


def test_add_datashader_gdf(point_gdf):
    m = EcoMap()
    img, bounds = datashade_gdf(point_gdf, "point")
    m.add_pil_image(img, bounds, zoom=False)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)


def test_add_datashader_gdf_with_zoom(poly_gdf):
    m = EcoMap()
    img, bounds = datashade_gdf(poly_gdf, "polygon")
    m.add_pil_image(img, bounds)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)
    assert m.view_state.longitude == (bounds[0] + bounds[2]) / 2
    assert m.view_state.latitude == (bounds[1] + bounds[3]) / 2
