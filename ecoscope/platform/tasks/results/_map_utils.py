from typing import Type

from pydantic import BaseModel

DEFAULT_TILE_LAYER_PRESETS = {
    "OpenStreetMap": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "title": "Open Street Map",
    },
    "ROADMAP": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "title": "Roadmap",
    },
    "SATELLITE": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "title": "Satellite",
    },
    "TERRAIN": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "title": "Terrain",
    },
    "USGS HILLSHADE": {
        "url": "https://server.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "title": "USGS Hillshade",
    },
}


def custom_tile_layer_json_schema(model_cls: Type[BaseModel]) -> dict:
    schema = model_cls.model_json_schema()
    schema["properties"]["url"]["title"] = "Custom Layer URL"
    schema["properties"]["opacity"]["title"] = "Custom Layer Opacity"
    schema["properties"]["max_zoom"]["title"] = "Custom Layer Max Zoom"
    schema["properties"]["min_zoom"]["title"] = "Custom Layer Min Zoom"
    schema["title"] = "Custom Layer (Advanced)"
    return schema


def preset_tile_layer_json_schema(model_cls: Type[BaseModel], preset_name: str, presets: dict) -> dict:
    schema = model_cls.model_json_schema()
    url = presets.get(preset_name, {}).get("url")
    title = presets.get(preset_name, {}).get("title")
    schema["properties"]["url"] |= {
        "const": url,
        "default": url,
        "enum": [url],
        "title": "Preset Layer URL",
    }
    schema["properties"]["url"].pop("description")
    schema["properties"]["url"].pop("pattern")
    schema["properties"].pop("max_zoom")
    schema["properties"].pop("min_zoom")
    schema["title"] = title
    return schema


def make_preset_or_custom_json_schema_extra(model_cls: Type[BaseModel], presets: dict):
    """Build a json_schema_extra callable that renders preset+custom variants for the given tile-layer model."""

    def _preset_or_custom_json_schema_extra(schema: dict) -> None:
        schema["items"]["title"] = "Base Layer"
        schema["items"]["anyOf"] = [
            preset_tile_layer_json_schema(model_cls, preset, presets) for preset in presets.keys()
        ]
        schema["items"]["anyOf"].append(custom_tile_layer_json_schema(model_cls))
        schema["default"] = [
            model_cls(layer_name="TERRAIN")._as_json_schema_default(),
            model_cls(layer_name="SATELLITE", opacity=0.5)._as_json_schema_default(),
        ]
        schema["ecoscope:advanced"] = True
        schema["items"].pop("$ref")

    return _preset_or_custom_json_schema_extra
