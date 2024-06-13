from ecoscope.mapping import (
    EcoMap,
    FloatElement,
    NorthArrowElement,
    ScaleElement,
    GeoTIFFElement,
    PrintControl,
)
import os
import ee
import geopandas
import pytest
from branca.element import MacroElement
from branca.colormap import StepColormap
from folium.raster_layers import TileLayer, ImageOverlay
from folium.map import FitBounds
from ecoscope.analysis.geospatial import datashade_gdf
from bs4 import BeautifulSoup


def test_ecomap_base():
    m = EcoMap()
    assert len(m._children) == 7


def test_repr_html():
    m = EcoMap(width=800, height=600)

    soup = BeautifulSoup(m._repr_html_(), "html.parser")
    assert soup.iframe.get("width") == "800"
    assert soup.iframe.get("height") == "600"

    soup = BeautifulSoup(m._repr_html_(fill_parent=True), "html.parser")
    assert soup.iframe.get("width") == "100%"
    assert soup.iframe.get("height") == "100%"

    assert m._parent.width == 800 and m._parent.height == 600


def test_static_map():
    m = EcoMap(width=800, height=600, static=True)
    assert len(m._children) == 2


def test_to_png():
    output_path = "tests/outputs/ecomap.png"
    m = EcoMap(width=800, height=600)
    m.to_png(output_path)
    assert os.path.exists(output_path)


def test_add_legend():
    m = EcoMap()
    # Test that we can assign hex colors with or without #
    m.add_legend(legend_dict={"Black_NoHash": "000000", "White_Hash": "#FFFFFF"})
    assert len(m.get_root()._children) == 2
    assert isinstance(list(m.get_root()._children.values())[1], MacroElement)
    html = m._repr_html_()
    assert "Black_NoHash" in html
    assert "White_Hash" in html


def test_add_north_arrow():
    m = EcoMap()
    m.add_north_arrow()
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], NorthArrowElement)
    assert "&lt;svg" in m._repr_html_()


def test_add_scale_bar():
    m = EcoMap()
    m.add_scale_bar()
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], ScaleElement)
    assert "&lt;svg" in m._repr_html_()


def test_add_title():
    m = EcoMap()
    m.add_title("THIS IS A TEST TITLE")
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], FloatElement)
    assert "THIS IS A TEST TITLE" in m._repr_html_()


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_image():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.Image("USGS/SRTMGL1_003")
    m.add_ee_layer(ee_object, vis_params, "DEM")
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], TileLayer)
    assert "earthengine.googleapis.com" in m._repr_html_()


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_image_collection():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5}
    ee_object = ee.ImageCollection("MODIS/006/MCD43C3")
    m.add_ee_layer(ee_object, vis_params, "DEM")
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], TileLayer)
    assert "earthengine.googleapis.com" in m._repr_html_()


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_add_ee_layer_feature_collection():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002/GEDI02_A_2021244154857_O15413_04_T05622_02_003_02_V002")
    m.add_ee_layer(ee_object, vis_params, "DEM")
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], TileLayer)
    assert "earthengine.googleapis.com" in m._repr_html_()


def test_zoom_to_bounds():
    m = EcoMap()
    m.zoom_to_bounds((34.683838, -3.173425, 38.869629, 0.109863))
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], FitBounds)
    assert "[[-3.173425, 34.683838], [0.109863, 38.869629]]" in m._repr_html_()


def test_zoom_to_gdf():
    m = EcoMap()
    x1 = 34.683838
    y1 = -3.173425
    x2 = 38.869629
    y2 = 0.109863
    gs = geopandas.GeoSeries.from_wkt(
        [f"POINT ({x1} {y1})", f"POINT ({x2} {y1})", f"POINT ({x1} {y2})", f"POINT ({x2} {y2})"]
    )
    gs = gs.set_crs("EPSG:4326")
    m.zoom_to_gdf(gs)
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], FitBounds)
    assert "[[-3.173425, 34.683838], [0.109863, 38.869629]]" in m._repr_html_()


def test_add_local_geotiff():
    m = EcoMap()
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap=None)
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], GeoTIFFElement)
    assert "new GeoRasterLayer" in m._repr_html_()


def test_add_local_geotiff_with_cmap():
    m = EcoMap()
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap="jet")
    assert len(m._children) == 9
    assert isinstance(list(m._children.values())[7], StepColormap)
    assert isinstance(list(m._children.values())[8], GeoTIFFElement)
    html = m._repr_html_()
    assert "new GeoRasterLayer" in html
    assert "d3.scale" in html


def test_add_print_control():
    m = EcoMap()
    m.add_print_control()
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], PrintControl)
    assert "L.control.BigImage()" in m._repr_html_()


@pytest.mark.parametrize(
    "file, geom_type",
    [
        ("tests/sample_data/vector/maec_4zones_UTM36S.gpkg", "polygon"),
        ("tests/sample_data/vector/observations.geojson", "point"),
    ],
)
def test_add_datashader_gdf(file, geom_type):
    m = EcoMap()
    gdf = geopandas.GeoDataFrame.from_file(file)
    img, bounds = datashade_gdf(gdf, geom_type)
    m.add_pil_image(img, bounds, zoom=False)
    assert len(m._children) == 8
    assert isinstance(list(m._children.values())[7], ImageOverlay)
    assert "L.imageOverlay(" in m._repr_html_()


def test_add_datashader_gdf_with_zoom():
    m = EcoMap()
    gdf = geopandas.GeoDataFrame.from_file("tests/sample_data/vector/maec_4zones_UTM36S.gpkg")
    img, bounds = datashade_gdf(gdf, "polygon")
    m.add_pil_image(img, bounds)
    assert len(m._children) == 9
    assert isinstance(list(m._children.values())[7], ImageOverlay)
    assert isinstance(list(m._children.values())[8], FitBounds)
    assert "L.imageOverlay(" in m._repr_html_()
