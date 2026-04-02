import random
from importlib.resources import files

import pytest
from pydantic import ValidationError
from wt_task import task

from ecoscope.platform.mock_loaders import load_parquet
from ecoscope.platform.tasks.results._ecomap import (
    LayerDefinition,
    LegendDefinition,
    LegendStyle,
    NorthArrowStyle,
    PointLayerStyle,
    PolylineLayerStyle,
    TextLayerStyle,
    TileLayer,
    ViewState,
    create_point_layer,
    create_polyline_layer,
    draw_ecomap,
    set_base_maps,
)


@pytest.fixture
def relocations():
    return load_parquet(files("ecoscope.platform.tasks.preprocessing") / "process-relocations.example-return.parquet")


@pytest.fixture
def trajectories():
    return load_parquet(
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )


@pytest.fixture
def events():
    return load_parquet(files("ecoscope.platform.tasks.io") / "get-events.example-return.parquet")


@pytest.fixture
def trajectories_colored(trajectories):
    # mock the output of 'apply_classification'
    trajectories["speed_bins"] = trajectories["speed_kmhr"].apply(lambda x: "Fast" if x > 0.5 else "Slow")
    # mock the output of 'apply_colormap'
    trajectories["colors"] = trajectories["speed_kmhr"].apply(
        lambda x: (255, 255, 0, 255) if x > 0.5 else (0, 255, 0, 255)
    )
    return trajectories


@pytest.fixture
def events_colored(events):
    # mock the output of 'apply_colormap'
    events["colors"] = events["event_category"].apply(
        lambda x: (255, 255, 0, 255) if x == "security" else (0, 255, 0, 255)
    )
    return events


def test_draw_ecomap_points(relocations):
    geo_layer = create_point_layer(
        geodataframe=relocations,
        layer_style=PointLayerStyle(get_radius=15, get_fill_color="#0000FF"),
    )
    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Relocations",
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_lines(trajectories):
    geo_layer = create_polyline_layer(
        geodataframe=trajectories,
        layer_style=PolylineLayerStyle(get_width=20, get_color="#00FFFF"),
    )

    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Trajectories",
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_combined(relocations, trajectories):
    relocs = create_point_layer(
        geodataframe=relocations,
        layer_style=PointLayerStyle(get_radius=20, get_color="#00FFFF"),
    )
    traj = create_polyline_layer(
        geodataframe=trajectories,
        layer_style=PolylineLayerStyle(get_width=20, get_color="#00FFFF"),
    )

    map_html = draw_ecomap(
        geo_layers=[relocs, traj],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Relocations and Trajectories",
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_with_colormap(trajectories_colored):
    geo_layer = create_polyline_layer(
        geodataframe=trajectories_colored,
        layer_style=PolylineLayerStyle(get_width=20, color_column="colors"),
    )

    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Trajectories",
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_with_legend(trajectories_colored):
    geo_layer = create_polyline_layer(
        geodataframe=trajectories_colored,
        layer_style=PolylineLayerStyle(get_width=20, color_column="colors"),
        legend=LegendDefinition(label_column="speed_bins", color_column="colors"),
    )

    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_with_legend_multiple_layers(relocations, trajectories_colored):
    relocations["labels"] = relocations["fixtime"].apply(
        lambda x: "new" if x > relocations["fixtime"].median() else "old"
    )
    relocations["colors"] = relocations["fixtime"].apply(
        lambda x: (255, 0, 0, 255) if x > relocations["fixtime"].median() else (0, 0, 255, 255)
    )
    layer1 = create_point_layer(
        geodataframe=relocations,
        layer_style=PointLayerStyle(get_radius=15, fill_color_column="colors"),
        legend=LegendDefinition(label_column="labels", color_column="colors"),
    )
    layer2 = create_polyline_layer(
        geodataframe=trajectories_colored,
        layer_style=PolylineLayerStyle(get_width=20, color_column="colors"),
        legend=LegendDefinition(label_column="speed_bins", color_column="colors"),
    )

    map_html = draw_ecomap(
        geo_layers=[layer1, layer2],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_widget_styles(trajectories_colored):
    geo_layer = create_polyline_layer(
        geodataframe=trajectories_colored,
        layer_style=PolylineLayerStyle(get_width=20, color_column="colors"),
        legend=LegendDefinition(label_column="speed_bins", color_column="colors"),
    )

    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Test",
        north_arrow_style=NorthArrowStyle(placement="top-left"),
        legend_style=LegendStyle(placement="top-right"),
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_combined_single_zoom(relocations, trajectories):
    relocs = create_polyline_layer(
        geodataframe=trajectories,
        layer_style=PolylineLayerStyle(get_width=20, get_color="#00FFFF"),
        zoom=True,
    )
    traj = create_polyline_layer(
        geodataframe=trajectories,
        layer_style=PolylineLayerStyle(get_width=20, get_color="#00FFFF"),
    )

    map_html = draw_ecomap(
        geo_layers=[relocs, traj],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Relocations and Trajectories",
    )
    assert isinstance(map_html, str)


def test_draw_ecomap_combined_view_state(relocations, trajectories):
    relocs = create_polyline_layer(
        geodataframe=trajectories,
        layer_style=PolylineLayerStyle(get_width=20, get_color="#00FFFF"),
        zoom=True,
    )
    traj = create_polyline_layer(
        geodataframe=trajectories,
        layer_style=PolylineLayerStyle(get_width=20, get_color="#00FFFF"),
    )

    map_html = draw_ecomap(
        geo_layers=[relocs, traj],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Relocations and Trajectories",
        view_state=ViewState(),
    )
    assert isinstance(map_html, str)


def test_view_state_validation(relocations):
    with pytest.raises(ValidationError):
        ViewState(
            longitude=20,
            latitude=180,
            zoom=15,
            pitch=2,
            bearing=12,
        )


def test_draw_ecomap_filters_null_geometry(events_colored):
    geo_layer = create_point_layer(
        geodataframe=events_colored,
        layer_style=PointLayerStyle(get_radius=20, color_column="colors"),
    )

    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Events",
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
        .partial(
            base_maps=[
                {"url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "opacity": 1},
                {
                    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    "opacity": 0.3,
                    "max_zoom": 17,
                    "min_zoom": None,
                },
            ]
        )
        .call()
    )

    assert isinstance(res[0], TileLayer)
    assert isinstance(res[1], TileLayer)


def test_draw_ecomap_with_textlayer(relocations):
    relocs = relocations.copy()
    if "fixtime" in relocs.columns:
        relocs["label"] = relocs["fixtime"].dt.strftime("%Y-%m-%d %H:%M")
    else:
        relocs["label"] = relocs.index.astype(str)

    text_layer_def = LayerDefinition(
        geodataframe=relocs,
        layer_style=TextLayerStyle(
            get_size=14,
            get_color=[0, 0, 0, 255],
            get_background_color=[255, 255, 255, 180],
            get_text_anchor="middle",
            get_alignment_baseline="center",
            font_weight="normal",
            pickable=False,
        ),
        legend=None,
        tooltip_columns=None,
        zoom=True,
    )

    map_html = draw_ecomap(
        geo_layers=[text_layer_def],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Relocation Labels",
    )

    assert isinstance(map_html, str)


def test_legend_style_title_format():
    title = "this_is_a_test"
    unformatted = LegendStyle(title=title)
    formatted = LegendStyle(title=title, format_title=True)

    assert unformatted.display_name == title
    assert formatted.display_name == "This Is A Test"


def test_draw_ecomap_points_with_radius(relocations):
    relocations["size"] = [random.randint(3, 8) for x in range(len(relocations))]
    geo_layer = create_point_layer(
        geodataframe=relocations,
        layer_style=PointLayerStyle(get_radius="size", get_fill_color="#0000FF"),
    )
    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Relocations",
    )

    assert isinstance(map_html, str)


def test_draw_ecomap_legend_sorts_numeric_strings_numerically(relocations):
    """
    Test that legend labels which are numeric strings sort numerically, not alphabetically.

    For example: "5", "10", "20" should sort as 5, 10, 20 (numeric)
    not as "10", "20", "5" (alphabetical).
    """
    # Create a subset with known numeric string labels that would sort differently
    # alphabetically vs numerically
    relocs = relocations.head(30).copy()

    # Assign numeric string labels: "5", "10", "20"
    # Alphabetically these would sort as: "10", "20", "5"
    # Numerically they should sort as: "5", "10", "20"
    labels = ["5", "10", "20"] * 10
    relocs["percentile"] = labels[: len(relocs)]

    # Assign colors for each label
    color_map = {
        "5": (0, 255, 0, 255),  # green
        "10": (255, 255, 0, 255),  # yellow
        "20": (255, 0, 0, 255),  # red
    }
    relocs["colors"] = relocs["percentile"].map(color_map)

    geo_layer = create_point_layer(
        geodataframe=relocs,
        layer_style=PointLayerStyle(get_radius=15, fill_color_column="colors"),
        legend=LegendDefinition(
            label_column="percentile",
            color_column="colors",
            sort="ascending",
        ),
    )

    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
        title="Numeric String Legend Sort Test",
    )

    assert isinstance(map_html, str)

    # Verify numeric sort order (5, 10, 20) appears in the HTML
    # The legend labels appear in the HTML in the order they were added
    idx_5 = map_html.find('"5"')
    idx_10 = map_html.find('"10"')
    idx_20 = map_html.find('"20"')

    # All labels should be present
    assert idx_5 != -1, "Label '5' not found in map HTML"
    assert idx_10 != -1, "Label '10' not found in map HTML"
    assert idx_20 != -1, "Label '20' not found in map HTML"

    # Labels should appear in numeric order: 5 before 10 before 20
    assert idx_5 < idx_10 < idx_20, (
        f"Labels not in numeric order. Positions: '5' at {idx_5}, '10' at {idx_10}, '20' at {idx_20}. "
        "Expected numeric sort order (5, 10, 20), not alphabetical (10, 20, 5)."
    )


def test_legend_shows_all_labels_with_shared_colors(trajectories):
    """
    Test that the legend shows all unique labels even when multiple labels
    share the same color. The legend should deduplicate by label (not by
    color), so each unique subject name gets its own entry.
    """
    traj = trajectories.head(30).copy()

    # Assign 3 subjects where two share the same color
    subjects = ["Subject A", "Subject B", "Subject C"] * 10
    traj["subject_name"] = subjects[: len(traj)]

    color_map = {
        "Subject A": (255, 255, 0, 255),  # yellow
        "Subject B": (255, 255, 0, 255),  # yellow (same as A)
        "Subject C": (0, 255, 0, 255),  # green
    }
    traj["subject_color"] = traj["subject_name"].map(color_map)

    geo_layer = create_polyline_layer(
        geodataframe=traj,
        layer_style=PolylineLayerStyle(get_width=5, color_column="subject_color"),
        legend=LegendDefinition(
            label_column="subject_name",
            color_column="subject_color",
        ),
    )

    map_html = draw_ecomap(
        geo_layers=[geo_layer],
        tile_layers=[TileLayer(layer_name="OpenStreetMap")],
    )

    assert isinstance(map_html, str)

    # All three subject names must appear in the legend,
    # even though Subject A and Subject B share the same color
    assert "Subject A" in map_html, "Legend missing 'Subject A'"
    assert "Subject B" in map_html, "Legend missing 'Subject B'"
    assert "Subject C" in map_html, "Legend missing 'Subject C'"
