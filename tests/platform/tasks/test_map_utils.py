import pytest
from pydantic import ValidationError

from ecoscope.platform.tasks.results._map_utils import (
    DEFAULT_TILE_LAYER_PRESETS,
    TileLayer,
    custom_tile_layer_json_schema,
    make_preset_or_custom_json_schema_extra,
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
