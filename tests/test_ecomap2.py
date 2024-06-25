import ee
import geopandas as gpd
import pytest
from ecoscope.mapping.lonboard_map import EcoMap2
from ecoscope.analysis.geospatial import datashade_gdf
from lonboard._layer import BitmapLayer, BitmapTileLayer, PolygonLayer
from lonboard._deck_widget import (
    NorthArrowWidget,
    ScaleWidget,
    TitleWidget,
    SaveImageWidget,
    FullscreenWidget,
)


def test_ecomap_base():
    m = EcoMap2()

    assert len(m.deck_widgets) == 3
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)
    assert isinstance(m.deck_widgets[0], FullscreenWidget)
    assert isinstance(m.deck_widgets[1], ScaleWidget)
    assert isinstance(m.deck_widgets[2], SaveImageWidget)


def test_static_map():
    m = EcoMap2(static=True)

    assert m.controller is False
    assert len(m.deck_widgets) == 1
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)
    assert isinstance(m.deck_widgets[0], ScaleWidget)


def test_add_legend():
    m = EcoMap2(default_widgets=False)
    m.add_legend(labels=["Black", "White"], colors=["#000000", "#FFFFFF"])
    assert len(m.deck_widgets) == 1


def test_add_north_arrow():
    m = EcoMap2()
    m.add_north_arrow()
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], NorthArrowWidget)


def test_add_scale_bar():
    m = EcoMap2()
    m.add_scale_bar()
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], ScaleWidget)


def test_add_title():
    m = EcoMap2()
    m.add_title("THIS IS A TEST TITLE")
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], TitleWidget)


def test_add_save_image():
    m = EcoMap2()
    m.add_save_image()
    assert len(m.deck_widgets) == 4
    assert isinstance(m.deck_widgets[3], SaveImageWidget)


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_image():
    m = EcoMap2()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.Image("USGS/SRTMGL1_003")
    m.add_ee_layer(ee_object, vis_params)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapTileLayer)


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_image_collection():
    m = EcoMap2()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5}
    ee_object = ee.ImageCollection("MODIS/006/MCD43C3")
    m.add_ee_layer(ee_object, vis_params)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapTileLayer)


@pytest.mark.skipif(not pytest.earthengine, reason="No onnection to EarthEngine.")
def test_add_ee_layer_feature_collection():
    m = EcoMap2()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002/GEDI02_A_2021244154857_O15413_04_T05622_02_003_02_V002")
    m.add_ee_layer(ee_object, vis_params)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapTileLayer)


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_geometry():
    m = EcoMap2()
    rectangle = ee.Geometry.Rectangle([-40, -20, 40, 20])
    m.add_ee_layer(rectangle, None)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], PolygonLayer)


def test_zoom_to_gdf():
    m = EcoMap2()
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
    m = EcoMap2()
    m.add_geotiff("tests/sample_data/raster/uint8.tif", cmap=None)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)


def test_add_geotiff_with_cmap():
    m = EcoMap2()
    m.add_geotiff("tests/sample_data/raster/uint8.tif", cmap="jet")
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)


@pytest.mark.parametrize(
    "file, geom_type",
    [
        ("tests/sample_data/vector/maec_4zones_UTM36S.gpkg", "polygon"),
        ("tests/sample_data/vector/observations.geojson", "point"),
    ],
)
def test_add_datashader_gdf(file, geom_type):
    m = EcoMap2()
    gdf = gpd.GeoDataFrame.from_file(file)
    img, bounds = datashade_gdf(gdf, geom_type)
    m.add_pil_image(img, bounds, zoom=False)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)


def test_add_datashader_gdf_with_zoom():
    m = EcoMap2()
    gdf = gpd.GeoDataFrame.from_file("tests/sample_data/vector/maec_4zones_UTM36S.gpkg")
    img, bounds = datashade_gdf(gdf, "polygon")
    m.add_pil_image(img, bounds)
    assert len(m.layers) == 2
    assert isinstance(m.layers[1], BitmapLayer)
    assert m.view_state.longitude == (bounds[0] + bounds[2]) / 2
    assert m.view_state.latitude == (bounds[1] + bounds[3]) / 2
