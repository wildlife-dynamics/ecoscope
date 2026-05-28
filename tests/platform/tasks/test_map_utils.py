import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from shapely.geometry import Point

from ecoscope.platform.tasks.results._map_utils import (
    DEFAULT_TILE_LAYER_PRESETS,
    TileLayer,
    custom_tile_layer_json_schema,
    make_preset_or_custom_json_schema_extra,
    persist_geoarrow_for_pydeck,
    preset_tile_layer_json_schema,
    set_base_maps,
)


def test_tile_layer_preset_resolves_url() -> None:
    tl = TileLayer(layer_name="TERRAIN")
    assert tl.url == DEFAULT_TILE_LAYER_PRESETS["TERRAIN"]["url"]


def test_tile_layer_unknown_preset_raises() -> None:
    with pytest.raises(ValidationError):
        TileLayer(layer_name="bogus")


def test_tile_layer_non_string_layer_name_falls_through() -> None:
    # passing a non-string value hits the `return v` fallthrough in the
    # before-mode field validator, after which the str type-check rejects it
    with pytest.raises(ValidationError):
        TileLayer.model_validate({"layer_name": 123})


def test_as_json_schema_default_has_expected_keys() -> None:
    payload = TileLayer(layer_name="TERRAIN")._as_json_schema_default()
    assert set(payload.keys()) == {"url", "opacity", "max_zoom", "min_zoom"}


def test_custom_tile_layer_json_schema_rewrites_titles() -> None:
    schema = custom_tile_layer_json_schema(TileLayer)
    assert schema["title"] == "Custom Layer (Advanced)"
    assert schema["properties"]["url"]["title"] == "Custom Layer URL"
    assert schema["properties"]["opacity"]["title"] == "Custom Layer Opacity"
    assert schema["properties"]["max_zoom"]["title"] == "Custom Layer Max Zoom"
    assert schema["properties"]["min_zoom"]["title"] == "Custom Layer Min Zoom"


def test_preset_tile_layer_json_schema_prunes_and_titles() -> None:
    schema = preset_tile_layer_json_schema(TileLayer, "TERRAIN", DEFAULT_TILE_LAYER_PRESETS)

    assert schema["title"] == DEFAULT_TILE_LAYER_PRESETS["TERRAIN"]["title"]
    assert schema["properties"]["url"]["const"] == DEFAULT_TILE_LAYER_PRESETS["TERRAIN"]["url"]
    assert "max_zoom" not in schema["properties"]
    assert "min_zoom" not in schema["properties"]
    assert "pattern" not in schema["properties"]["url"]
    assert "description" not in schema["properties"]["url"]


def test_make_preset_or_custom_json_schema_extra_mutates_schema() -> None:
    extra = make_preset_or_custom_json_schema_extra(TileLayer, DEFAULT_TILE_LAYER_PRESETS)
    schema: dict = {"items": {"$ref": "#/$defs/TileLayer"}}

    extra(schema)

    assert schema["items"]["title"] == "Base Layer"
    assert "$ref" not in schema["items"]
    assert isinstance(schema["items"]["anyOf"], list)
    assert len(schema["items"]["anyOf"]) == len(DEFAULT_TILE_LAYER_PRESETS) + 1
    assert schema["ecoscope:advanced"] is True
    assert len(schema["default"]) == 2


def test_set_base_maps_default_returns_two_layers() -> None:
    layers = set_base_maps(None)
    assert len(layers) == 2
    assert all(isinstance(layer, TileLayer) for layer in layers)


def _points_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"val": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        crs="EPSG:4326",
    )


def test_persist_geoarrow_for_pydeck_roundtrip(tmp_path) -> None:
    gdf = _points_gdf()
    dst = persist_geoarrow_for_pydeck(gdf, str(tmp_path), filename="demo")

    assert dst.endswith("demo.parquet")
    schema = pq.read_schema(dst)
    # GeoArrow stores points as a struct<x, y> tagged with the geoarrow.point extension.
    geom_field = schema.field("geometry")
    assert geom_field.metadata[b"ARROW:extension:name"] == b"geoarrow.point"
    gdf_read = gpd.read_parquet(dst)
    assert list(gdf_read["val"]) == [1, 2, 3]
    assert gdf_read.crs == gdf.crs


def test_persist_geoarrow_for_pydeck_re_encodes_columns(tmp_path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "val": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
            "ts_naive": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ts_aware": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
            "rgba": [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)],
            "rgb": [(10, 20, 30), (40, 50, 60), (70, 80, 90)],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    dst = persist_geoarrow_for_pydeck(gdf, str(tmp_path), filename="reencoded")
    schema = pq.read_schema(dst)
    assert schema.field("val").type == pa.float32()
    assert schema.field("ts_naive").type == pa.string()
    assert schema.field("ts_aware").type == pa.string()
    assert schema.field("rgba").type == pa.list_(pa.uint8(), 4)
    assert schema.field("rgb").type == pa.list_(pa.uint8(), 3)


def test_persist_geoarrow_for_pydeck_color_column_with_nulls_raises(tmp_path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "rgba": [(255, 0, 0, 255), None, (0, 0, 255, 255)],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="null values"):
        persist_geoarrow_for_pydeck(gdf, str(tmp_path), filename="colors")


def test_persist_geoarrow_for_pydeck_skips_non_color_object_columns(tmp_path) -> None:
    # Ragged tuples must not be coerced to a fixed-size color list.
    gdf = gpd.GeoDataFrame(
        {
            "ragged": [(1, 2), (3, 4, 5), (6, 7)],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )
    dst = persist_geoarrow_for_pydeck(gdf, str(tmp_path), filename="ragged")
    schema = pq.read_schema(dst)
    # Left untouched as an object column → arrow encodes it however pandas defaults
    # (list/struct/binary), but it must NOT be uint8 fixed-size-list.
    assert schema.field("ragged").type != pa.list_(pa.uint8(), 3)
    assert schema.field("ragged").type != pa.list_(pa.uint8(), 4)
