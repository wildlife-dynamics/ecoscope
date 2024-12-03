import os
import random
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import ecoscope
import shapely
from ecoscope.mapping import EcoMap
from ecoscope.analysis.geospatial import datashade_gdf
from ecoscope.analysis.classifier import apply_classification, apply_color_map
from lonboard._layer import BitmapLayer, BitmapTileLayer, PathLayer, PolygonLayer, ScatterplotLayer
from lonboard._deck_widget import (
    NorthArrowWidget,
    ScaleWidget,
    TitleWidget,
    SaveImageWidget,
    FullscreenWidget,
    LegendWidget,
)


@pytest.fixture
def poly_gdf():
    gdf = gpd.GeoDataFrame.from_file("tests/sample_data/vector/maec_4zones_UTM36S.gpkg")
    return gdf


@pytest.fixture
def line_gdf():
    gdf = pd.read_csv("tests/sample_data/vector/KDB025Z.csv", index_col="id")
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
    assert len(m.layers) == 0
    assert isinstance(m.deck_widgets[0], FullscreenWidget)
    assert isinstance(m.deck_widgets[1], ScaleWidget)
    assert isinstance(m.deck_widgets[2], SaveImageWidget)


def test_static_map():
    m = EcoMap(static=True)

    assert m.controller is False
    assert len(m.deck_widgets) == 1
    assert len(m.layers) == 0
    assert isinstance(m.deck_widgets[0], ScaleWidget)


def test_add_legend():
    m = EcoMap(default_widgets=False)
    m.add_legend(labels=["Black", "White"], colors=["#000000", "#FFFFFF"])
    assert len(m.deck_widgets) == 1


def test_add_legend_tuples():
    m = EcoMap(default_widgets=False)
    m.add_legend(labels=["Black", "White"], colors=[(0, 0, 0, 255), (255, 255, 255, 255)])
    assert len(m.deck_widgets) == 1


def test_add_legend_mixed():
    m = EcoMap(default_widgets=False)
    m.add_legend(labels=["Black", "White"], colors=[(0, 0, 0, 255), "#FFFFFF"])
    assert len(m.deck_widgets) == 1


def test_add_legend_series():
    m = EcoMap(default_widgets=False)
    m.add_legend(labels=pd.Series(["Black", "White"]), colors=pd.Series([(0, 0, 0, 255), (255, 255, 255, 255)]))
    assert len(m.deck_widgets) == 1
    legend = m.deck_widgets[0]
    assert isinstance(legend, LegendWidget)
    assert legend.labels == ["Black", "White"]
    assert legend.colors == ["rgba(0, 0, 0, 1.0)", "rgba(255, 255, 255, 1.0)"]


def test_add_legend_series_with_nan():
    m = EcoMap(default_widgets=False)
    m.add_legend(
        labels=pd.Series([0, 1, np.nan, 5, np.nan]),
        colors=pd.Series([(0, 0, 0, 255), (255, 255, 255, 255), (0, 0, 0, 0), (100, 100, 100, 255), (0, 0, 0, 0)]),
    )
    assert len(m.deck_widgets) == 1
    legend = m.deck_widgets[0]
    assert isinstance(legend, LegendWidget)
    assert legend.labels == ["0.0", "1.0", "5.0"]
    assert legend.colors == ["rgba(0, 0, 0, 1.0)", "rgba(255, 255, 255, 1.0)", "rgba(100, 100, 100, 1.0)"]


def test_add_legend_series_unbalanced_good():
    m = EcoMap(default_widgets=False)
    m.add_legend(
        labels=pd.Series(["Black", "White", "White"]),
        colors=pd.Series([(0, 0, 0, 255), (255, 255, 255, 255), (255, 255, 255, 255)]),
    )


def test_add_legend_series_unbalanced_bad():
    m = EcoMap(default_widgets=False)
    with pytest.raises(ValueError, match="Unique label and color values must be of equal number"):
        m.add_legend(labels=pd.Series(["Black", "White"]), colors=pd.Series([(0, 0, 0, 255), (0, 0, 0, 255)]))


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


@pytest.mark.io
def test_add_ee_layer_image():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.Image("USGS/SRTMGL1_003")
    m.add_layer(EcoMap.ee_layer(ee_object, vis_params))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)


@pytest.mark.io
def test_add_ee_layer_image_collection():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5}
    ee_object = ee.ImageCollection("MODIS/006/MCD43C3")
    m.add_layer(EcoMap.ee_layer(ee_object, vis_params))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)
    assert m.layers[0].tile_size == 256


@pytest.mark.io
def test_add_ee_layer_feature_collection():
    m = EcoMap()
    vis_params = {"min": 0, "max": 4000, "opacity": 0.5, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
    ee_object = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002/GEDI02_A_2021244154857_O15413_04_T05622_02_003_02_V002")
    m.add_layer(EcoMap.ee_layer(ee_object, vis_params))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)


@pytest.mark.io
def test_add_ee_layer_geometry():
    m = EcoMap()
    rectangle = ee.Geometry.Rectangle([-40, -20, 40, 20])
    m.add_layer(EcoMap.ee_layer(rectangle, None))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], PolygonLayer)


def test_add_polyline(line_gdf):
    m = EcoMap()
    m.add_layer(m.polyline_layer(line_gdf, get_width=200))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], PathLayer)
    assert m.layers[0].get_width == 200


def test_add_point(point_gdf):
    m = EcoMap()
    m.add_layer(m.point_layer(point_gdf, get_fill_color=[25, 100, 25, 100]))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], ScatterplotLayer)
    # default color
    assert m.layers[0].get_fill_color == [25, 100, 25, 100]


def test_add_polygon(poly_gdf):
    m = EcoMap()
    m.add_layer(m.polygon_layer(poly_gdf, extruded=True, get_line_width=35), zoom=True)
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], PolygonLayer)
    assert m.layers[0].extruded
    assert m.layers[0].get_line_width == 35
    # validating zoom param by checking view state is non-default
    assert m.view_state.longitude != 10
    assert m.view_state.latitude != 0


def test_layers_from_gdf(poly_gdf, line_gdf, point_gdf):
    joint_kwargs = {
        "get_width": 130,
        "get_line_width": 35,
        "get_radius": 200,
        "get_fill_color": [25, 100, 25, 100],
        "get_bananas": 2134,
    }

    poly_gdf.to_crs(4326, inplace=True)
    point_gdf.to_crs(4326, inplace=True)

    together = gpd.GeoDataFrame(pd.concat([poly_gdf.geometry, line_gdf.geometry, point_gdf.geometry]))
    layers = EcoMap.layers_from_gdf(gdf=together, **joint_kwargs)

    m = EcoMap(layers=layers)
    assert len(m.layers) == 3
    assert m.layers[0].get_fill_color == [25, 100, 25, 100]
    assert m.layers[0].get_line_width == 35
    assert m.layers[1].get_width == 130
    assert m.layers[2].get_radius == 200
    assert m.layers[2].get_fill_color == [25, 100, 25, 100]


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
    assert m.view_state.zoom == 6

    m.zoom_to_bounds(feat=gpd.GeoDataFrame(geometry=gs), max_zoom=1)
    assert m.view_state.longitude == (x1 + x2) / 2
    assert m.view_state.latitude == (y1 + y2) / 2
    assert m.view_state.zoom == 1


def test_geotiff_layer():
    m = EcoMap()
    m.add_layer(EcoMap.geotiff_layer("tests/sample_data/raster/uint8.tif", cmap=None))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapLayer)


def test_geotiff_layer_with_cmap():
    m = EcoMap()
    m.add_layer(EcoMap.geotiff_layer("tests/sample_data/raster/uint8.tif", cmap="jet"))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapLayer)


def test_geotiff_layer_in_mem_with_cmap():
    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "maec_4zones_UTM36S.gpkg"))

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.unary_union, in_crs=AOI.crs, out_crs=AOI.crs, xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    grid["fake_density"] = grid.apply(lambda _: random.randint(1, 50), axis=1)
    raster = ecoscope.io.raster.grid_to_raster(
        grid,
        val_column="fake_density",
    )

    m = EcoMap()
    m.add_layer(EcoMap.geotiff_layer(raster, cmap="jet"))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapLayer)


def test_add_datashader_gdf(point_gdf):
    m = EcoMap()
    img, bounds = datashade_gdf(point_gdf, "point")
    m.add_layer(EcoMap.pil_layer(img, bounds))
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapLayer)


def test_add_datashader_gdf_with_zoom(poly_gdf):
    m = EcoMap()
    img, bounds = datashade_gdf(poly_gdf, "polygon")
    m.add_layer(EcoMap.pil_layer(img, bounds), zoom=True)
    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapLayer)
    assert m.view_state.longitude == (bounds[0] + bounds[2]) / 2
    assert m.view_state.latitude == (bounds[1] + bounds[3]) / 2


def test_add_polyline_with_color(movebank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movebank_relocations)
    # this is effectively a reimplementation of SpeedDataFrame
    apply_classification(
        trajectory,
        input_column_name="speed_kmhr",
        output_column_name="speed_bins",
        scheme="equal_interval",
        label_suffix=" km/h",
        label_ranges=True,
        k=6,
    )
    cmap = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027"]
    apply_color_map(trajectory, "speed_bins", cmap=cmap, output_column_name="speed_colors")

    m = EcoMap()
    m.add_layer(m.polyline_layer(trajectory, color_column="speed_colors", get_width=2000))
    m.add_legend(labels=trajectory["speed_bins"], colors=trajectory["speed_colors"])

    assert len(m.layers) == 1
    assert isinstance(m.layers[0], PathLayer)
    assert m.layers[0].get_width == 2000


def test_add_point_with_color(point_gdf):
    point_gdf["time"] = point_gdf["recorded_at"].apply(lambda x: x.value)
    apply_classification(point_gdf, input_column_name="time", scheme="equal_interval")
    apply_color_map(point_gdf, "time_classified", "viridis", output_column_name="time_cmap")

    m = EcoMap()
    m.add_layer(m.point_layer(point_gdf, fill_color_column="time_cmap", get_radius=10000))

    assert len(m.layers) == 1
    assert isinstance(m.layers[0], ScatterplotLayer)


def test_add_polygon_with_color(poly_gdf):
    apply_color_map(poly_gdf, "ZoneID", "tab20b")

    m = EcoMap()
    m.add_layer(
        m.polygon_layer(poly_gdf, fill_color_column="ZoneID_colormap", extruded=True, get_line_width=35), zoom=True
    )

    assert len(m.layers) == 1
    assert isinstance(m.layers[0], PolygonLayer)
    assert m.layers[0].extruded
    assert m.layers[0].get_line_width == 35
    # validating zoom param by checking view state is non-default
    assert m.view_state.longitude != 10
    assert m.view_state.latitude != 0


def test_add_named_tile_layer():
    m = EcoMap()
    m.add_layer(m.get_named_tile_layer("TERRAIN", opacity=0.3))

    assert len(m.layers) == 1
    assert isinstance(m.layers[0], BitmapTileLayer)
    assert m.layers[0].opacity == 0.3
