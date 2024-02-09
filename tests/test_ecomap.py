from ecoscope.mapping import EcoMap
import ee
import geopandas
import json


def test_ecomap_base():
    EcoMap()


def test_add_legend():
    m = EcoMap()
    # Test that we can assign hex colors with or without #
    m.add_legend(legend_dict={"Black_NoHash": "000000", "White_Hash": "#FFFFFF"})
    html = m._repr_html_()
    assert "Black_NoHash" in html
    assert "White_Hash" in html


def test_add_north_arrow():
    m = EcoMap()
    m.add_north_arrow()
    html = m._repr_html_()
    assert "&lt;svg" in html


def test_add_scale_bar():
    m = EcoMap()
    m.add_scale_bar()
    html = m._repr_html_()
    assert "&lt;svg" in html


def test_add_title():
    m = EcoMap()
    m.add_title("THIS IS A TEST TITLE")
    html = m._repr_html_()
    assert "THIS IS A TEST TITLE" in html


def test_add_ee_layer():
    m = EcoMap()
    ee.Initialize()
    with open("./tests/sample_data/ee/eeImage.json") as f:
        img = ee.deserializer.decode(json.load(f))
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    m.add_ee_layer(img, vis_params, "DEM")
    html = m._repr_html_()
    assert "earthengine.googleapis.com" in html


def test_zoom_to_bounds():
    m = EcoMap()
    m.zoom_to_bounds((34.683838, -3.173425, 38.869629, 0.109863))
    html = m._repr_html_()
    assert "[[-3.173425, 34.683838], [0.109863, 38.869629]]" in html


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
    html = m._repr_html_()
    assert "[[-3.173425, 34.683838], [0.109863, 38.869629]]" in html


def test_add_local_geotiff():
    m = EcoMap()
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap=None)
    html = m._repr_html_()
    assert "new GeoRasterLayer" in html


def test_add_local_geotiff_with_cmap():
    m = EcoMap()
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap="jet")
    html = m._repr_html_()
    assert "new GeoRasterLayer" in html
    assert "d3.scale" in html


def test_add_print_control():
    m = EcoMap()
    m.add_print_control()
    html = m._repr_html_()
    assert "L.control.BigImage()" in html
