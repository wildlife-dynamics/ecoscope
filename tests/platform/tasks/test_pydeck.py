import json
import os
from pathlib import Path
from typing import Literal

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pydeck as pdk  # type: ignore[import-untyped, import-not-found]
import pytest
from pydantic import ValidationError
from shapely.geometry import MultiPolygon, Point, Polygon
from wt_task import task

from ecoscope.platform.tasks.results._map_utils import set_base_maps
from ecoscope.platform.tasks.results._pydeck import (
    BitmapLayerDefinition,
    DeckJsonSpec,
    GeoJSONLayerStyle,
    HexagonLayerStyle,
    IconLayerStyle,
    LegendFromDataframe,
    LegendSegment,
    LegendStyle,
    LegendValue,
    PathLayerStyle,
    PolygonLayerStyle,
    PydeckLayerDefinition,
    ScatterplotLayerStyle,
    TextLayerStyle,
    TileLayer,
    ViewState,
    _model_dump_with_pydeck_literals,
    create_geojson_layer,
    create_hexagon_layer,
    create_icon_layer,
    create_path_layer,
    create_polygon_layer_pydeck,
    create_scatterplot_layer,
    create_text_layer_pydeck,
    create_tiled_bitmap_layer,
    draw_map,
    merge_tile_layers,
    rewrite_file_urls_for_screenshots,
    view_state_from_geodataframes,
    view_state_from_layers,
)

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "sample_data" / "vector"


@pytest.fixture
def gdf_with_points():
    gdf = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_point.geojson"))
    gdf["color"] = gdf["category"].apply(lambda x: (255, 255, 0, 255) if x == "first" else (0, 255, 0, 255))
    return gdf


def test_geojson_layer():
    gdf = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_poly.geojson"))
    layer_def = create_geojson_layer(
        geodataframe=gdf,
        layer_style=GeoJSONLayerStyle(get_fill_color=[255, 0, 0]),
    )

    map_html = draw_map(
        geo_layers=[layer_def],
    )
    assert isinstance(map_html, str)


def test_path_layer():
    gdf = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_path.geojson"))
    layer_def = create_path_layer(
        geodataframe=gdf,
        layer_style=PathLayerStyle(get_width=3, get_color=[255, 0, 0]),
    )

    map_html = draw_map(
        geo_layers=[layer_def],
    )
    assert isinstance(map_html, str)


def test_scatterplot_layer(
    gdf_with_points,
):
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius=500, get_fill_color="color"),
    )

    map_html = draw_map(
        geo_layers=[layer_def],
    )
    assert isinstance(map_html, str)


def test_scatterplot_radius_from_column(gdf_with_points):
    gdf_with_points["size"] = list(range(3, 3 + len(gdf_with_points)))

    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius="size"),
    )

    np.testing.assert_array_equal(
        layer_def.layer_style.get_radius,
        gdf_with_points["size"].values,
    )


def test_scatterplot_radius_lifts_negatives(gdf_with_points):
    # min raw value is -4, so post-lift min must be 1 and the gap of 1 between
    # consecutive values must be preserved.
    gdf_with_points["size"] = list(range(-4, -4 + len(gdf_with_points)))

    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius="size"),
    )

    normalized_radius_values = np.asarray(layer_def.layer_style.get_radius)
    assert normalized_radius_values.min() == 1
    np.testing.assert_array_equal(normalized_radius_values, np.arange(1, 1 + len(gdf_with_points)))


def test_scatterplot_radius_handles_nans(gdf_with_points):
    # NaN must collapse to 1 while real values are bumped by 1 to offset.
    sizes = [float(3 + i) for i in range(len(gdf_with_points))]  # [3, 4, 5, 6, ...]
    sizes[0] = float("nan")
    sizes[2] = float("nan")
    gdf_with_points["size"] = sizes

    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius="size"),
    )

    normalized_radius_values = np.asarray(layer_def.layer_style.get_radius)
    assert normalized_radius_values[0] == 1  # was NaN
    assert normalized_radius_values[1] == 5  # was 4
    assert normalized_radius_values[2] == 1  # was NaN
    assert normalized_radius_values[3] == 7  # was 6
    assert not np.isnan(normalized_radius_values).any()


def test_scatterplot_radius_handles_negatives_and_nans(gdf_with_points):
    # Combined: negatives get lifted so min real value is 1, then everything
    # gets +1 (real values now >= 2) and NaNs are filled with 1.
    sizes = [float(v) for v in range(-2, -2 + len(gdf_with_points))]
    sizes[0] = float("nan")
    gdf_with_points["size"] = sizes

    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius="size"),
    )

    normalized_radius_values = np.asarray(layer_def.layer_style.get_radius)
    assert normalized_radius_values[0] == 1  # was NaN
    assert normalized_radius_values.min() == 1
    real_min = np.nanmin(normalized_radius_values[1:])
    assert real_min == 2
    assert not np.isnan(normalized_radius_values).any()


def test_scatterplot_radius_numeric_passthrough(gdf_with_points):
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius=500),
    )
    assert layer_def.layer_style.get_radius == 500


def test_hexagon_layer(gdf_with_points):
    layer_def = create_hexagon_layer(
        geodataframe=gdf_with_points,
        layer_style=HexagonLayerStyle(
            radius=5000,
            extruded=True,
            auto_highlight=True,
            elevation_scale=500,  # Make hexagons tall enough to see
        ),
    )

    map_html = draw_map(
        geo_layers=[layer_def],
        tile_layers=[
            TileLayer(
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            )
        ],
    )

    assert isinstance(map_html, str)


def test_polygon_layer():
    gdf = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_poly.geojson"))
    layer_def = create_polygon_layer_pydeck(
        geodataframe=gdf,
        layer_style=PolygonLayerStyle(get_fill_color=[255, 0, 0]),
    )

    map_html = draw_map(
        geo_layers=[layer_def],
    )
    assert isinstance(map_html, str)


def test_draw_map_explodes_multipolygon():
    """draw_map should explode MultiPolygon geometries into Polygons for deck.gl compatibility."""
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    multi = MultiPolygon([poly1, poly2])
    gdf = gpd.GeoDataFrame({"name": ["test"]}, geometry=[multi], crs="EPSG:4326")

    assert gdf.geometry.iloc[0].geom_type == "MultiPolygon"

    layer_def = create_polygon_layer_pydeck(
        geodataframe=gdf,
        layer_style=PolygonLayerStyle(get_fill_color=[255, 0, 0]),
    )

    map_html = draw_map(geo_layers=[layer_def])
    assert isinstance(map_html, str)

    # Verify the HTML contains Polygon (not MultiPolygon) geometry type
    assert '"type": "Polygon"' in map_html
    assert '"type": "MultiPolygon"' not in map_html


def test_text_layer():
    gdf = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_text.geojson"))
    layer_def = create_text_layer_pydeck(geodataframe=gdf, layer_style=TextLayerStyle(get_text="text"))
    map_html = draw_map(
        geo_layers=[layer_def],
    )
    assert isinstance(map_html, str)


def test_icon_layer():
    gdf = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_icon.geojson"))
    gdf["icon_data"] = gdf["icon_data"].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
    layer_def = create_icon_layer(geodataframe=gdf, layer_style=IconLayerStyle(get_icon="icon_data", get_size=60))
    map_html = draw_map(
        geo_layers=[layer_def],
    )
    assert isinstance(map_html, str)


@pytest.mark.parametrize("extruded", [True, False])
def test_combined_with_extrusion(extruded):
    lines = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_path.geojson"))
    path_layer_def = create_path_layer(
        geodataframe=lines,
        layer_style=PathLayerStyle(get_width=3, get_color=[255, 0, 0]),
    )
    polys = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_poly.geojson"))
    poly_layer_def = create_polygon_layer_pydeck(
        geodataframe=polys,
        layer_style=PolygonLayerStyle(
            get_fill_color=[255, 0, 0],
            extruded=extruded,
        ),
    )

    map_html = draw_map(
        geo_layers=[path_layer_def, poly_layer_def],
    )
    assert isinstance(map_html, str)
    expected_depth_test = "true" if extruded else "false"
    assert f'"depthTest": {expected_depth_test}' in map_html


def test_draw_map_json_output():
    lines = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_path.geojson"))
    path_layer_def = create_path_layer(
        geodataframe=lines,
        layer_style=PathLayerStyle(get_width=3, get_color=[255, 0, 0]),
    )
    polys = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_poly.geojson"))
    poly_layer_def = create_polygon_layer_pydeck(
        geodataframe=polys,
        layer_style=PolygonLayerStyle(get_fill_color=[255, 0, 0]),
    )

    view_state = ViewState(longitude=10.5, latitude=-2.3, zoom=8, pitch=15, bearing=45)

    spec = draw_map(
        geo_layers=[path_layer_def, poly_layer_def],
        view_state=view_state,
        title="My Map",
        output_type="json",
    )

    assert isinstance(spec, DeckJsonSpec)

    assert spec.initialViewState == {
        "longitude": 10.5,
        "latitude": -2.3,
        "zoom": 8,
        "pitch": 15,
        "bearing": 45,
    }

    assert len(spec.layers) == 2
    layer_types_by_id = {layer["id"]: layer["@@type"] for layer in spec.layers}
    assert layer_types_by_id == {"PathLayer-0": "PathLayer", "PolygonLayer-1": "PolygonLayer"}

    widget_types = {w["@@type"] for w in spec.widgets}
    assert {"NorthArrowWidget", "ScaleWidget", "SaveImageWidget", "TitleWidget"} <= widget_types

    assert isinstance(spec.views, dict)
    assert spec.views["@@type"] == "MapView"
    assert spec.views["controller"] is True
    assert spec.views["repeat"] is True

    north_arrow = next(w for w in spec.widgets if w["@@type"] == "NorthArrowWidget")
    assert north_arrow["placement"] == "top-left"

    # extras (mapStyle, mapProvider, parameters) flow through via extra="allow"
    dumped = spec.model_dump()
    assert {"layers", "initialViewState", "views", "widgets", "parameters", "mapStyle"} <= dumped.keys()


def test_draw_map_json_validation_rejects_bad_shape():
    """DeckJsonSpec rejects payloads missing required top-level keys."""
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        DeckJsonSpec.model_validate({"layers": []})  # missing initialViewState + views


def test_view_state_calc():
    x1 = 34.683838
    y1 = -3.173425
    x2 = 38.869629
    y2 = 0.109863
    gs = gpd.GeoSeries.from_wkt(
        [
            f"POINT ({x1} {y1})",
            f"POINT ({x2} {y1})",
            f"POINT ({x1} {y2})",
            f"POINT ({x2} {y2})",
        ]
    )
    gs = gs.set_crs("EPSG:4326")

    vs = view_state_from_geodataframes(geodataframes=[gpd.GeoDataFrame(geometry=gs)])
    assert vs.longitude == (x1 + x2) / 2
    assert vs.latitude == (y1 + y2) / 2
    assert vs.zoom == 6

    vs = view_state_from_geodataframes(geodataframes=[gpd.GeoDataFrame(geometry=gs)], max_zoom=2)
    assert vs.longitude == (x1 + x2) / 2
    assert vs.latitude == (y1 + y2) / 2
    assert vs.zoom == 2


def test_pydeck_literals():
    path = _model_dump_with_pydeck_literals(PathLayerStyle())
    point = _model_dump_with_pydeck_literals(ScatterplotLayerStyle())
    poly = _model_dump_with_pydeck_literals(PolygonLayerStyle())

    assert isinstance(path["width_units"], pdk.types.String)
    assert isinstance(point["line_width_units"], pdk.types.String)
    assert isinstance(poly["line_width_units"], pdk.types.String)


@pytest.mark.parametrize(
    "label_column, color_column, sort, suffix",
    [
        ("category", "color", "descending", None),
        ("category", "color", "ascending", None),
        ("category", "color", None, None),
        ("category", "missing", "descending", None),
        ("missing", "color", "descending", None),
        ("category", "color", "descending", "_suffix!!!"),
    ],
)
def test_build_legend_from_dataframe(
    label_column: str,
    color_column: str,
    sort: Literal["ascending", "descending"] | None,
    suffix: str,
    gdf_with_points,
):
    legend_definition = LegendFromDataframe(
        label_column=label_column,
        color_column=color_column,
        sort=sort,
        label_suffix=suffix,
        title="An Legend",
    )
    if label_column == "missing" or color_column == "missing":
        with pytest.raises(KeyError):
            legend_definition.build_legend_from_dataframe(gdf_with_points)
    else:
        actual_legend = legend_definition.build_legend_from_dataframe(gdf_with_points)
        expected_legend = LegendSegment(
            title="An Legend",
            values=[
                LegendValue(
                    label="first" if not suffix else f"first{suffix}",
                    color="rgba(255, 255, 0, 1.0)",
                ),
                LegendValue(
                    label="second" if not suffix else f"second{suffix}",
                    color="rgba(0, 255, 0, 1.0)",
                ),
            ],
        )
        if sort == "descending":
            expected_legend.values.reverse()

        assert actual_legend == expected_legend


def test_build_legend_from_multiple_dataframe(gdf_with_points):
    point_layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius=500, get_fill_color="color"),
        legend=LegendFromDataframe(
            label_column="category",
            color_column="color",
            sort="descending",
            label_suffix=" SUFFIX",
            title="Points",
        ),
    )

    lines = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_path.geojson"))
    lines["color"] = lines["category"].apply(lambda x: (100, 100, 100, 255) if x == "first" else (20, 255, 600, 255))
    path_layer_def = create_path_layer(
        geodataframe=lines,
        layer_style=PathLayerStyle(get_width=3, get_color="color"),
        legend=LegendFromDataframe(
            label_column="category",
            color_column="color",
            title="Lines",
        ),
    )
    map_html = draw_map(
        geo_layers=[path_layer_def, point_layer_def],
    )
    assert isinstance(map_html, str)


def test_layer_with_legend_from_dataframe(gdf_with_points):
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius=500, get_fill_color="color"),
        legend=LegendFromDataframe(
            label_column="category",
            color_column="color",
            sort="descending",
            label_suffix=" SUFFIX",
        ),
    )

    map_html = draw_map(
        geo_layers=[layer_def],
    )
    assert isinstance(map_html, str)


def test_custom_legend(gdf_with_points):
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius=500, get_fill_color="color"),
        legend=LegendSegment(
            title="An Legend",
            values=[
                LegendValue(label="Very Bad", color="#FF0000"),
                LegendValue(label="Very Good", color="#00FF00"),
            ],
        ),
    )

    map_html = draw_map(geo_layers=[layer_def])
    assert isinstance(map_html, str)


def test_legends_are_combined_across_layers(gdf_with_points):
    scatter_layer = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_radius=500, get_fill_color="color"),
        legend=LegendSegment(
            title="Scatter Legend",
            values=[LegendValue(label="Points", color="#00FF00")],
        ),
    )
    polys = gpd.read_file(os.path.join(TEST_DATA_DIR, "test_poly.geojson"))
    polygon_layer = create_polygon_layer_pydeck(
        geodataframe=polys,
        layer_style=PolygonLayerStyle(get_fill_color=[255, 0, 0]),
        legend=LegendSegment(title="Polygon Legend", values=[LegendValue(label="Polys", color="#FF0000")]),
    )

    map_html = draw_map(
        geo_layers=[scatter_layer, polygon_layer],
    )
    assert "Scatter Legend" in map_html
    assert "Polygon Legend" in map_html


def test_legend_style_format_title_display_name():
    style = LegendStyle()
    formatted = LegendStyle(format_title=True)

    title = "this_is_a_test"
    assert style.display_name(title) == title
    assert formatted.display_name(title) == "This Is A Test"


def test_legend_style_format_title_applied_in_draw_map(gdf_with_points):
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_fill_color="color"),
        legend=LegendSegment(
            title="my_layer_legend",
            values=[LegendValue(label="A", color="#FF0000")],
        ),
    )

    raw_html = draw_map(geo_layers=[layer_def])
    assert "my_layer_legend" in raw_html
    assert "My Layer Legend" not in raw_html

    formatted_html = draw_map(
        geo_layers=[layer_def],
        legend_style=LegendStyle(format_title=True),
    )
    assert "My Layer Legend" in formatted_html
    assert "my_layer_legend" not in formatted_html


def test_legend_style_format_title_does_not_mutate_input(gdf_with_points):
    """format_title must not mutate user-provided LegendSegments."""
    legend = LegendSegment(
        title="my_legend",
        values=[LegendValue(label="A", color="#FF0000")],
    )
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_fill_color="color"),
        legend=legend,
    )

    draw_map(
        geo_layers=[layer_def],
        legend_style=LegendStyle(format_title=True),
    )
    assert legend.title == "my_legend"


def test_legend_from_dataframe_sorts_numeric_strings_numerically(gdf_with_points):
    """Numeric string labels should sort numerically, not alphabetically.

    Alphabetic sort: "10", "2", "5"
    Numeric sort:    "2", "5", "10"
    """
    n = len(gdf_with_points)
    labels = ["2", "5", "10"]
    gdf_with_points["pct"] = [labels[i % 3] for i in range(n)]
    color_map = {"2": (0, 255, 0, 255), "5": (255, 255, 0, 255), "10": (255, 0, 0, 255)}
    gdf_with_points["pct_color"] = gdf_with_points["pct"].map(color_map)

    legend_def = LegendFromDataframe(
        label_column="pct",
        color_column="pct_color",
        sort="ascending",
        title="Percentile",
    )

    segment = legend_def.build_legend_from_dataframe(gdf_with_points)
    sorted_labels = [v.label for v in segment.values]
    assert sorted_labels == ["2", "5", "10"]


def test_legend_from_dataframe_sort_descending_numeric_strings(gdf_with_points):
    n = len(gdf_with_points)
    labels = ["2", "5", "10"]
    gdf_with_points["pct"] = [labels[i % 3] for i in range(n)]
    color_map = {"2": (0, 255, 0, 255), "5": (255, 255, 0, 255), "10": (255, 0, 0, 255)}
    gdf_with_points["pct_color"] = gdf_with_points["pct"].map(color_map)

    legend_def = LegendFromDataframe(
        label_column="pct",
        color_column="pct_color",
        sort="descending",
        title="Percentile",
    )

    segment = legend_def.build_legend_from_dataframe(gdf_with_points)
    sorted_labels = [v.label for v in segment.values]
    assert sorted_labels == ["10", "5", "2"]


def test_legend_from_dataframe_sort_falls_back_to_string_for_non_numeric(gdf_with_points):
    """Non-numeric labels should still sort lexicographically (pd.to_numeric leaves them alone)."""
    legend_def = LegendFromDataframe(
        label_column="category",
        color_column="color",
        sort="ascending",
        title="Cat",
    )
    segment = legend_def.build_legend_from_dataframe(gdf_with_points)
    sorted_labels = [v.label for v in segment.values]
    assert sorted_labels == ["first", "second"]


def test_map_with_title():
    map_html = draw_map(
        geo_layers=[],
        title="A title",
    )

    assert isinstance(map_html, str)


def test_map_with_tiles():
    map_html = draw_map(
        geo_layers=[],
        tile_layers=[
            TileLayer(
                url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                max_zoom=3,
            )
        ],
        widget_id="12345",
    )

    assert isinstance(map_html, str)


def test_tile_layer_name_string_validator():
    TileLayer(layer_name="OpenStreetMap", opacity=1.0)

    with pytest.raises(ValidationError):
        TileLayer(layer_name="not a real layer", opacity=1.0)


def test_custom_tile_layer_validation():
    TileLayer(
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        max_zoom=13,
    )

    with pytest.raises(ValidationError):
        TileLayer(
            url="ftp://test/?give=tiles_pls",
        )


def test_set_base_maps():
    res = (
        task(set_base_maps)
        .validate()
        .call(
            base_maps=[
                {"url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "opacity": 1},
                {
                    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    "opacity": 0.3,
                    "max_zoom": 3,
                    "min_zoom": 0,
                },
            ]
        )
    )
    assert isinstance(res[0], TileLayer)
    assert isinstance(res[1], TileLayer)


def test_geojson_layer_numpy_types():
    """
    Verify that GeoJsonLayer correctly handles numpy data types (int64, float32, etc.)
    during serialization. This ensures that columns with numpy types in the
    GeoDataFrame don't cause JSON serialization errors when pydeck converts them.
    """

    # Create a GDF with explicit numpy types that often cause serialization issues
    df = pd.DataFrame(
        {
            "int64_col": np.array([1, 2], dtype=np.int64),
            "float32_col": np.array([1.1, 2.2], dtype=np.float32),
            "object_numpy_col": [np.int64(10), np.float64(20.5)],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

    layer_def = create_geojson_layer(
        geodataframe=gdf,
    )
    map_html = draw_map(geo_layers=[layer_def])
    assert isinstance(map_html, str)


def test_geojson_layer_styling():
    """
    Verify that static styling properties (like filled, stroked, colors, line widths)
    are correctly passed from GeoJSONLayerStyle to the generated deck.gl layer.
    """
    gdf = gpd.GeoDataFrame({"col": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    style = GeoJSONLayerStyle(
        filled=False,
        stroked=True,
        get_line_color=[0, 255, 0],
        get_line_width=10,
        line_width_min_pixels=2,
        extruded=True,
        elevation_scale=5,
        get_elevation=100,
    )
    layer_def = create_geojson_layer(geodataframe=gdf, layer_style=style)

    map_html = draw_map(geo_layers=[layer_def])

    assert isinstance(map_html, str)
    # Verify that key properties are present in the generated HTML
    # Pydeck converts snake_case to camelCase for JS props
    assert "getLineColor" in map_html
    assert "getLineWidth" in map_html
    assert "lineWidthMinPixels" in map_html
    assert "elevationScale" in map_html
    assert "getElevation" in map_html


def test_geojson_layer_data_driven_styling():
    """
    Verify that string accessors (referencing dataframe columns) are correctly
    passed to pydeck using the default '@@=colname' syntax. This confirms that
    data-driven styling works by referencing properties in the GeoJSON features.
    """
    df = pd.DataFrame(
        {
            "fill_color": [[255, 0, 0, 255]],
            "line_color": [[0, 255, 0, 255]],
            "line_width": [5],
            "elevation": [100],
            "radius": [20],
            "geometry": [Point(0, 0)],
        }
    )
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

    style = GeoJSONLayerStyle(
        get_fill_color="fill_color",
        get_line_color="line_color",
        get_line_width="line_width",
        get_elevation="elevation",
        get_point_radius="radius",
        # Add some static props to ensure mixing works
        extruded=True,
        filled=True,
    )

    layer_def = create_geojson_layer(geodataframe=gdf, layer_style=style)
    map_html = draw_map(geo_layers=[layer_def])

    # Remove whitespace for robust string matching
    compact_html = "".join(map_html.split())

    # Check for the correct JS accessor syntax
    assert '"getFillColor":"@@=fill_color"' in compact_html
    assert '"getLineColor":"@@=line_color"' in compact_html
    assert '"getLineWidth":"@@=line_width"' in compact_html
    assert '"getElevation":"@@=elevation"' in compact_html
    assert '"getPointRadius":"@@=radius"' in compact_html

    # Check that static properties are still there
    assert '"extruded":true' in compact_html


def test_geojson_layer_mixed_styling():
    """
    Verify that mixing static values (e.g., fixed color) and data-driven accessors
    (e.g., variable elevation from a column) works correctly in the same layer style.
    """
    # Create a simple polygon dataframe
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    df = pd.DataFrame({"elevation_val": [500], "geometry": [poly]})
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

    style = GeoJSONLayerStyle(
        get_fill_color=[0, 0, 255, 128],  # Static blue color
        get_elevation="elevation_val",  # Data-driven elevation
        extruded=True,
        filled=True,
    )

    layer_def = create_geojson_layer(geodataframe=gdf, layer_style=style)
    map_html = draw_map(geo_layers=[layer_def])

    # Remove whitespace
    compact_html = "".join(map_html.split())

    # Assertions
    # Static color should be a literal array
    assert '"getFillColor":[0,0,255,128]' in compact_html
    # Data-driven elevation should be an accessor
    assert '"getElevation":"@@=elevation_val"' in compact_html
    # Static boolean flags
    assert '"extruded":true' in compact_html
    assert '"filled":true' in compact_html


# Tests for TileLayer, BitmapLayerDefinition, and create_tiled_bitmap_layer


def test_tiled_bitmap_layer_definition_defaults():
    """Test that TileLayer has correct defaults."""
    tile_def = TileLayer(url="https://example.com/{z}/{x}/{y}.png")
    assert tile_def.url == "https://example.com/{z}/{x}/{y}.png"
    assert tile_def.opacity == 1.0
    assert tile_def.max_zoom == 20
    assert tile_def.min_zoom == 0


def test_tiled_bitmap_layer_definition_custom_values():
    """Test TileLayer with custom values."""
    tile_def = TileLayer(
        url="https://example.com/{z}/{x}/{y}.png",
        opacity=0.7,
        max_zoom=15,
        min_zoom=5,
    )
    assert tile_def.opacity == 0.7
    assert tile_def.max_zoom == 15
    assert tile_def.min_zoom == 5


def test_create_tiled_bitmap_layer_basic():
    """Test create_tiled_bitmap_layer returns a TileLayer."""
    result = create_tiled_bitmap_layer(
        url="https://earthengine.googleapis.com/v1/tiles/{z}/{x}/{y}",
    )
    assert isinstance(result, TileLayer)
    assert result.url == "https://earthengine.googleapis.com/v1/tiles/{z}/{x}/{y}"
    assert result.opacity == 1.0


def test_create_tiled_bitmap_layer_with_options():
    """Test create_tiled_bitmap_layer with custom options."""
    result = create_tiled_bitmap_layer(
        url="https://example.com/tiles/{z}/{x}/{y}",
        opacity=0.8,
        max_zoom=18,
        min_zoom=2,
    )
    assert result.opacity == 0.8
    assert result.max_zoom == 18
    assert result.min_zoom == 2


def test_draw_map_with_bitmap_overlay():
    """Test draw_map with BitmapLayerDefinition in tile_layers."""
    # Create a simple geo layer
    gdf = gpd.GeoDataFrame(
        {"col": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    geo_layer = create_scatterplot_layer(geodataframe=gdf)

    # Create bitmap overlay tile layer
    overlay = BitmapLayerDefinition(
        image="https://example.com/raster/image.png",
        bounds=[-1, -1, 1, 1],
        opacity=0.9,
    )

    map_html = draw_map(
        geo_layers=[geo_layer],
        tile_layers=[overlay],
    )

    assert isinstance(map_html, str)
    assert "https://example.com/raster/image.png" in map_html


def test_draw_map_mixed_tile_layers():
    """Test draw_map with both TileLayer and BitmapLayerDefinition."""
    gdf = gpd.GeoDataFrame(
        {"col": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    geo_layer = create_scatterplot_layer(geodataframe=gdf)

    base_tile = TileLayer(
        url="https://base.example.com/{z}/{x}/{y}.png",
    )
    overlay = BitmapLayerDefinition(
        image="https://overlay.example.com/image.png",
        bounds=[-1, -1, 1, 1],
    )

    map_html = draw_map(
        geo_layers=[geo_layer],
        tile_layers=[base_tile, overlay],
    )

    assert isinstance(map_html, str)
    assert "https://base.example.com" in map_html
    assert "https://overlay.example.com" in map_html


def test_draw_map_multiple_bitmap_overlays():
    """Test draw_map with multiple BitmapLayerDefinition overlays."""
    gdf = gpd.GeoDataFrame(
        {"col": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    geo_layer = create_scatterplot_layer(geodataframe=gdf)

    overlays = [
        BitmapLayerDefinition(
            image="https://overlay1.example.com/image1.png",
            bounds=[-1, -1, 1, 1],
            opacity=0.5,
        ),
        BitmapLayerDefinition(
            image="https://overlay2.example.com/image2.png",
            bounds=[-2, -2, 2, 2],
            opacity=0.8,
        ),
    ]

    map_html = draw_map(
        geo_layers=[geo_layer],
        tile_layers=overlays,
    )

    assert isinstance(map_html, str)
    assert "https://overlay1.example.com" in map_html
    assert "https://overlay2.example.com" in map_html


def test_draw_map_bitmap_only_no_geo_layers():
    """Test draw_map with only tile layers, no geo_layers."""
    overlay = BitmapLayerDefinition(
        image="https://raster-only.example.com/image.png",
        bounds=[-1, -1, 1, 1],
    )
    base_tile = TileLayer(
        url="https://base.example.com/{z}/{x}/{y}.png",
    )

    map_html = draw_map(
        geo_layers=None,
        tile_layers=[base_tile, overlay],
    )

    assert isinstance(map_html, str)
    assert "https://raster-only.example.com" in map_html
    assert "https://base.example.com" in map_html


def test_draw_map_bitmap_overlay_with_legend():
    """Test draw_map renders legend from BitmapLayerDefinition."""
    legend = LegendSegment(
        values=[
            {"label": "0.10", "color": "#440154"},
            {"label": "0.50", "color": "#21918c"},
            {"label": "0.90", "color": "#fde725"},
        ],
        title="NDVI",
    )
    overlay = BitmapLayerDefinition(
        image="https://raster.example.com/image.png",
        bounds=[-1, -1, 1, 1],
        legend=legend,
    )

    map_html = draw_map(
        tile_layers=[overlay],
    )

    assert isinstance(map_html, str)
    assert "LegendWidget" in map_html
    assert "NDVI" in map_html


def test_bitmap_layer_definition_with_legend():
    """Test BitmapLayerDefinition accepts and stores legend."""
    legend = LegendSegment(
        values=[
            {"label": "low", "color": "#000000"},
            {"label": "high", "color": "#ffffff"},
        ],
        title="Test",
    )
    bld = BitmapLayerDefinition(
        image="https://example.com/image.png",
        bounds=[-1, -1, 1, 1],
        legend=legend,
    )
    assert bld.legend is not None
    assert bld.legend.title == "Test"
    assert len(bld.legend.values) == 2


def test_bitmap_layer_definition_legend_defaults_none():
    """Test BitmapLayerDefinition legend defaults to None."""
    bld = BitmapLayerDefinition(
        image="https://example.com/image.png",
        bounds=[-1, -1, 1, 1],
    )
    assert bld.legend is None


def test_legend_shows_all_labels_with_shared_colors(gdf_with_points):
    """
    Test that the legend shows all unique labels even when multiple labels
    share the same color. The legend should deduplicate by label (not by
    color), so each unique name gets its own entry.
    """
    extra = gpd.GeoDataFrame(
        {"category": ["third"], "color": [(255, 255, 0, 255)]},
        geometry=[Point(0, 0)],
        crs=gdf_with_points.crs,
    )
    gdf = pd.concat([gdf_with_points, extra], ignore_index=True)

    legend_def = LegendFromDataframe(
        label_column="category",
        color_column="color",
        title="Subject",
    )
    legend_segment = legend_def.build_legend_from_dataframe(gdf)

    # All three labels must appear, even though "first" and "third" share a color
    labels = [v.label if hasattr(v, "label") else v["label"] for v in legend_segment.values]
    assert "first" in labels, "Legend missing 'first'"
    assert "second" in labels, "Legend missing 'second'"
    assert "third" in labels, "Legend missing 'third'"


# Tests for GeoJSON URL support


def test_create_geojson_layer_with_url():
    """create_geojson_layer with data_url sets data_url and leaves geodataframe None."""
    url = "https://example.com/data.geojson"
    layer_def = create_geojson_layer(data_url=url)

    assert isinstance(layer_def, PydeckLayerDefinition)
    assert layer_def.data_url == url
    assert layer_def.geodataframe is None
    assert layer_def.layer_type == "GeoJsonLayer"


def test_layer_definition_rejects_file_url():
    """PydeckLayerDefinition rejects file:// data_urls; only http(s) URLs are allowed."""
    with pytest.raises(ValueError, match="file://"):
        create_geojson_layer(data_url="file:///some/long/path/data.geojson")


def test_draw_map_with_url_layer():
    """draw_map with a URL-backed layer embeds the URL in the generated HTML."""
    url = "https://example.com/data.geojson"
    layer_def = create_geojson_layer(data_url=url)

    map_html = draw_map(geo_layers=[layer_def])

    assert isinstance(map_html, str)
    assert url in map_html


def test_view_state_from_layers_url_only():
    """view_state_from_layers with a URL-only layer returns a default ViewState without error."""
    url_layer = create_geojson_layer(data_url="https://example.com/data.geojson")

    # Should not raise even though no geodataframe is present
    vs = view_state_from_layers(layers=[url_layer])
    assert vs.longitude == 0
    assert vs.latitude == 0


def test_view_state_from_layers_mixed_url_and_gdf(gdf_with_points):
    """view_state_from_layers with mixed layers computes bounds only from the GDF layer."""
    url_layer = create_geojson_layer(data_url="https://example.com/data.geojson")
    gdf_layer = create_scatterplot_layer(geodataframe=gdf_with_points)

    vs = view_state_from_layers(layers=[url_layer, gdf_layer])
    # The view state should be non-default (computed from gdf_layer's data)
    assert vs.zoom > 0


def test_draw_map_combined_single_zoom(gdf_with_points):
    flagged = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_fill_color=[255, 0, 0]),
        zoom=True,
    )
    other = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_fill_color=[0, 255, 0]),
    )

    map_html = draw_map(
        geo_layers=[flagged, other],
        title="Combined Single Zoom",
    )
    assert isinstance(map_html, str)
    stripped_html = "".join(map_html.split())
    assert '"initialViewState":{"bearing":0,"latitude":-1.4,"longitude":35.155,"pitch":0,"zoom":10.0}' in stripped_html


def test_draw_map_combined_view_state(gdf_with_points):
    flagged = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_fill_color=[255, 0, 0]),
        zoom=True,
    )
    other = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(get_fill_color=[0, 255, 0]),
    )

    map_html = draw_map(
        geo_layers=[flagged, other],
        title="Combined View State",
        view_state=ViewState(),
    )
    assert isinstance(map_html, str)
    stripped_html = "".join(map_html.split())
    assert '"initialViewState":{"bearing":0,"latitude":0,"longitude":0,"pitch":0,"zoom":0}' in stripped_html


def test_draw_map_url_layer_with_legend_from_dataframe_skipped():
    """LegendFromDataframe is skipped (with a warning) for URL-only layers (no GDF)."""
    url = "https://example.com/data.geojson"
    legend = LegendFromDataframe(label_column="category", color_column="color")
    layer_def = create_geojson_layer(data_url=url, legend=legend)

    # Should not raise; LegendFromDataframe is skipped since geodataframe is None
    map_html = draw_map(geo_layers=[layer_def])

    assert isinstance(map_html, str)
    # No LegendWidget should be present since the only legend was skipped
    assert "LegendWidget" not in map_html


def test_draw_map_url_layer_with_legend_segment():
    """LegendSegment (static) works fine for URL-only layers."""
    url = "https://example.com/data.geojson"
    legend = LegendSegment(
        title="My Legend",
        values=[LegendValue(label="Feature", color="#FF0000")],
    )
    layer_def = create_geojson_layer(data_url=url, legend=legend)

    map_html = draw_map(geo_layers=[layer_def])

    assert isinstance(map_html, str)
    assert "My Legend" in map_html


# Tests for merge_tile_layers


def test_merge_tile_layers_both_none():
    result = merge_tile_layers(base_layers=None, overlay=None)
    assert result == []


def test_merge_tile_layers_base_only():
    base = TileLayer(url="https://base.example.com/{z}/{x}/{y}.png")
    result = merge_tile_layers(base_layers=[base], overlay=None)
    assert result == [base]


def test_merge_tile_layers_overlay_only():
    overlay = BitmapLayerDefinition(
        image="https://overlay.example.com/image.png",
        bounds=[-1, -1, 1, 1],
    )
    result = merge_tile_layers(base_layers=None, overlay=overlay)
    assert result == [overlay]


def test_merge_tile_layers_both():
    base1 = TileLayer(url="https://base1.example.com/{z}/{x}/{y}.png")
    base2 = TileLayer(url="https://base2.example.com/{z}/{x}/{y}.png")
    overlay = BitmapLayerDefinition(
        image="https://overlay.example.com/image.png",
        bounds=[-1, -1, 1, 1],
    )
    result = merge_tile_layers(base_layers=[base1, base2], overlay=overlay)
    assert result == [base1, base2, overlay]


def test_merge_tile_layers_order():
    """Base layers come before overlay in the merged list."""
    base = TileLayer(url="https://base.example.com/{z}/{x}/{y}.png")
    overlay = BitmapLayerDefinition(
        image="https://overlay.example.com/image.png",
        bounds=[-1, -1, 1, 1],
    )
    result = merge_tile_layers(base_layers=[base], overlay=overlay)
    assert isinstance(result[0], TileLayer)
    assert isinstance(result[1], BitmapLayerDefinition)


# Tests for rewrite_file_urls_for_screenshots


def test_rewrite_file_urls_replaces_url():
    file_url = "file:///some/long/path/data.geojson"
    html = f'<script>url = "{file_url}"</script>'
    result = rewrite_file_urls_for_screenshots(html=html, file_urls=[file_url])
    assert file_url not in result
    assert "http://127.0.0.1:8099/data.geojson" in result


def test_rewrite_file_urls_preserves_basename_only():
    file_url = "file:///very/deep/nested/directory/mydata.json"
    html = f'"{file_url}"'
    result = rewrite_file_urls_for_screenshots(html=html, file_urls=[file_url])
    assert "mydata.json" in result
    assert "nested/directory" not in result


def test_rewrite_file_urls_multiple():
    url1 = "file:///path/a.geojson"
    url2 = "file:///path/b.geojson"
    html = f'"{url1}" and "{url2}"'
    result = rewrite_file_urls_for_screenshots(html=html, file_urls=[url1, url2])
    assert url1 not in result
    assert url2 not in result
    assert "a.geojson" in result
    assert "b.geojson" in result


def test_rewrite_file_urls_custom_port(monkeypatch):
    monkeypatch.setenv("ECOSCOPE_SCREENSHOT_FILE_SERVER_PORT", "9000")
    file_url = "file:///path/data.geojson"
    html = f'"{file_url}"'
    result = rewrite_file_urls_for_screenshots(html=html, file_urls=[file_url])
    assert "http://127.0.0.1:9000/data.geojson" in result


def test_rewrite_file_urls_empty_list():
    html = "<html>no urls</html>"
    result = rewrite_file_urls_for_screenshots(html=html, file_urls=[])
    assert result == html


# Tests for TooltipWidget assembly


def test_draw_map_tooltip_widget_added_for_pickable_layer(gdf_with_points):
    """TooltipWidget is added when any geo layer is pickable (default)."""
    layer_def = create_scatterplot_layer(geodataframe=gdf_with_points)
    map_html = draw_map(geo_layers=[layer_def])
    assert "TooltipWidget" in map_html


def test_draw_map_tooltip_widget_omitted_when_no_pickable_layer(gdf_with_points):
    """TooltipWidget is omitted when no geo layer is pickable."""
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        layer_style=ScatterplotLayerStyle(pickable=False),
    )
    map_html = draw_map(geo_layers=[layer_def])
    assert "TooltipWidget" not in map_html


def test_draw_map_tooltip_widget_omitted_with_no_geo_layers():
    """TooltipWidget is omitted when there are no geo layers."""
    map_html = draw_map(geo_layers=[])
    assert "TooltipWidget" not in map_html


def test_draw_map_tooltip_widget_default_layer_columns_empty(gdf_with_points):
    """Default rendering ships an empty layerColumns dict — JS side will fall back to 'show all'."""
    layer_def = create_scatterplot_layer(geodataframe=gdf_with_points)
    map_html = draw_map(geo_layers=[layer_def])
    compact_html = "".join(map_html.split())
    assert "TooltipWidget" in compact_html
    assert '"layerColumns":{}' in compact_html


def test_draw_map_tooltip_widget_layer_columns_keyed_by_layer_id(gdf_with_points):
    """tooltip_columns surfaces in the widget layerColumns dict keyed by f'{layer_type}-{index}'."""
    layer_def = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        tooltip_columns=["name", "category"],
    )
    map_html = draw_map(geo_layers=[layer_def])
    compact_html = "".join(map_html.split())
    assert '"ScatterplotLayer-0":["name","category"]' in compact_html


def test_draw_map_tooltip_widget_layer_columns_skips_layers_with_none(gdf_with_points):
    """A layer with tooltip_columns=None is omitted from layerColumns (= 'show all' on JS side)."""
    explicit = create_scatterplot_layer(
        geodataframe=gdf_with_points,
        tooltip_columns=["category"],
    )
    default = create_path_layer(geodataframe=gpd.read_file(os.path.join(TEST_DATA_DIR, "test_path.geojson")))
    map_html = draw_map(geo_layers=[explicit, default])
    compact_html = "".join(map_html.split())
    assert '"ScatterplotLayer-0":["category"]' in compact_html
    assert "PathLayer-1" not in compact_html.split('"layerColumns":')[1].split("}")[0]


def test_draw_map_assigns_explicit_layer_ids(gdf_with_points):
    """Each geo layer gets f'{layer_type}-{index}' so the tooltip widget can address them."""
    a = create_scatterplot_layer(geodataframe=gdf_with_points)
    b = create_scatterplot_layer(geodataframe=gdf_with_points)
    map_html = draw_map(geo_layers=[a, b])
    compact_html = "".join(map_html.split())
    assert '"id":"ScatterplotLayer-0"' in compact_html
    assert '"id":"ScatterplotLayer-1"' in compact_html
