import re
from typing import Annotated, Any, Type

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField

OpacityAnnotation = Annotated[
    float,
    AdvancedField(
        title="Layer Opacity",
        description="Set layer transparency from 1 (fully visible) to 0 (hidden).",
        default=1,
        ge=0,
        le=1,
    ),
]


@register()
def set_layer_opacity(
    opacity: OpacityAnnotation = 1,
) -> float:
    return opacity


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


class TileLayer(BaseModel):
    """A tiled raster layer loaded on-demand from a URL template (e.g., base maps)."""

    layer_name: SkipJsonSchema[str] = ""
    url: Annotated[
        str,
        Field(
            default="https://example.tiles.com/{z}/{x}/{y}.png",
            title="Layer URL",
            pattern=re.compile(
                r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}([-a-zA-Z0-9()@:%_\+.~#?&//=\{\}]*)"
            ),
            description="The URL of a publicly accessible tiled raster service.",
        ),
    ] = "https://example.tiles.com/{z}/{x}/{y}.png"
    opacity: OpacityAnnotation = 1
    max_zoom: Annotated[
        int,
        Field(
            default=20,
            title="Layer Max Zoom",
            description="Set the maximum zoom level to fetch tiles for.",
        ),
    ] = 20
    min_zoom: Annotated[
        int,
        Field(
            default=0,
            title="Layer Min Zoom",
            description="Set the minimum zoom level to fetch tiles for.",
        ),
    ] = 0

    @field_validator("layer_name", mode="before")
    def _tile_layer_name_from_string(v: Any):
        if isinstance(v, str):
            for layer_name in DEFAULT_TILE_LAYER_PRESETS.keys():
                if layer_name == v:
                    return v
            raise ValueError(
                f"String input must match one of: {[layer_name for layer_name in DEFAULT_TILE_LAYER_PRESETS.keys()]}"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def set_url(cls, values):
        if (
            isinstance(values, dict)
            and values.get("layer_name")
            and values.get("layer_name") != ""
            and values.get("layer_name") in DEFAULT_TILE_LAYER_PRESETS.keys()
        ):
            values["url"] = DEFAULT_TILE_LAYER_PRESETS.get(values.get("layer_name")).get("url")
        return values

    def _as_json_schema_default(self):
        return {
            "url": self.url,
            "opacity": self.opacity,
            "max_zoom": self.max_zoom,
            "min_zoom": self.min_zoom,
        }


def custom_tile_layer_json_schema(tile_layer: TileLayer) -> dict:
    schema = tile_layer.model_json_schema()
    schema["properties"]["url"]["title"] = "Custom Layer URL"
    schema["properties"]["opacity"]["title"] = "Custom Layer Opacity"
    schema["properties"]["max_zoom"]["title"] = "Custom Layer Max Zoom"
    schema["properties"]["min_zoom"]["title"] = "Custom Layer Min Zoom"
    schema["title"] = "Custom Layer (Advanced)"
    return schema


def preset_tile_layer_json_schema(tile_layer: TileLayer, preset_name: str, presets: dict) -> dict:
    schema = tile_layer.model_json_schema()
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


def make_preset_or_custom_json_schema_extra(tile_layer: TileLayer, presets: dict):
    """Build a json_schema_extra callable that renders preset+custom variants for the given tile-layer model."""

    def _preset_or_custom_json_schema_extra(schema: dict) -> None:
        schema["items"]["title"] = "Base Layer"
        schema["items"]["anyOf"] = [
            preset_tile_layer_json_schema(tile_layer, preset, presets) for preset in presets.keys()
        ]
        schema["items"]["anyOf"].append(custom_tile_layer_json_schema(tile_layer))
        schema["default"] = [
            tile_layer(layer_name="TERRAIN")._as_json_schema_default(),
            tile_layer(layer_name="SATELLITE", opacity=0.5)._as_json_schema_default(),
        ]
        schema["ecoscope:advanced"] = True
        schema["items"].pop("$ref")

    return _preset_or_custom_json_schema_extra


_preset_or_custom_json_schema_extra = make_preset_or_custom_json_schema_extra(TileLayer, DEFAULT_TILE_LAYER_PRESETS)


@register()
def set_base_maps(
    base_maps: Annotated[
        list[TileLayer] | SkipJsonSchema[None],
        Field(
            json_schema_extra=_preset_or_custom_json_schema_extra,
            title=" ",
            description=(
                "Select tile layers to use as base layers in map outputs."
                " The first layer in the list will be the bottommost layer displayed."
            ),
        ),
    ] = None,
) -> Annotated[list[TileLayer], Field()]:
    if base_maps is None:
        base_maps = [
            TileLayer(layer_name="TERRAIN"),
            TileLayer(layer_name="SATELLITE", opacity=0.5),
        ]
    return base_maps
