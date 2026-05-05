import logging
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Tuple, TypeAlias, Union, no_type_check

logger = logging.getLogger(__name__)

import pandas as pd
import pydeck as pdk  # type: ignore[import-untyped, import-not-found]
from pydantic import BaseModel, Field, PlainSerializer, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.tasks.results._map_utils import (
    DEFAULT_TILE_LAYER_PRESETS,
    make_preset_or_custom_json_schema_extra,
)

TileLayerPresets = DEFAULT_TILE_LAYER_PRESETS

PYDECK_CUSTOM_LIBRARIES = [
    {
        "libraryName": "EcoscopeDeckglExtensions",
        "resourceUri": "https://cdn.jsdelivr.net/npm/@ecoscope/ecoscope-deckgl-extensions@0.0.7/dist/bundle.js",
    }
]

# Wraps string values in pdk.types.String at dump time so pydeck treats them
# as literal strings rather than data accessor expressions.
_pdk_literal_string = PlainSerializer(lambda v: pdk.types.String(v), when_used="unless-none")
PydeckString = Annotated[str, _pdk_literal_string]

UnitType = Annotated[Literal["meters", "pixels"], _pdk_literal_string]
WidgetPlacement = Annotated[
    Literal["top-left", "top-right", "bottom-left", "bottom-right", "fill"],
    _pdk_literal_string,
]
AlignmentBaseline = Annotated[Literal["top", "center", "bottom"], _pdk_literal_string]
TextAnchor = Annotated[Literal["start", "middle", "end"], _pdk_literal_string]
WordBreak = Annotated[Literal["break-word", "break-all"], _pdk_literal_string]
ColorAccessor = str | SkipJsonSchema[list[int]] | SkipJsonSchema[list[list[int]]]
FloatAccessor = str | float | SkipJsonSchema[list[float]]
LegendLabel: TypeAlias = str
LegendColor: TypeAlias = str


class LayerStyleBase(BaseModel):
    auto_highlight: Annotated[bool, AdvancedField(default=False)] = False
    opacity: Annotated[float, AdvancedField(default=1, ge=0, le=1)] = 1
    pickable: Annotated[bool, AdvancedField(default=True)] = True


class HexagonLayerStyle(LayerStyleBase):
    """
    Hexagon Layer style kwargs
    See https://deck.gl/docs/api-reference/aggregation-layers/hexagon-layer for more info
    """

    get_position: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    radius: Annotated[float, AdvancedField(default=1000)] = 1000
    color_aggregation: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_color_weight: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    extruded: Annotated[bool, AdvancedField(default=False)] = False
    wireframe: Annotated[bool, AdvancedField(default=True)] = True
    elevation_scale: Annotated[float, AdvancedField(default=1)] = 1
    elevation_aggregation: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    coverage: Annotated[float, AdvancedField(default=1, le=1, ge=0)] = 1


class PathLayerStyle(LayerStyleBase):
    """
    Path Layer style kwargs
    See https://deck.gl/docs/api-reference/layers/path-layer for more info
    """

    get_path: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=3)] = 3
    width_scale: Annotated[float, AdvancedField(default=1)] = 1
    width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    width_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    cap_rounded: Annotated[bool, AdvancedField(default=True)] = True
    joint_rounded: Annotated[bool, AdvancedField(default=False)] = False
    billboard: Annotated[bool, AdvancedField(default=False)] = False


class ScatterplotLayerStyle(LayerStyleBase):
    """
    Scatterplot Layer style kwargs
    See https://deck.gl/docs/api-reference/layers/scatterplot-layer for more info
    """

    get_position: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_fill_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_line_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_radius: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=5)] = 5
    get_line_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=1)] = 1
    get_elevation: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    radius_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    radius_scale: Annotated[float, AdvancedField(default=1)] = 1
    radius_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    radius_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    line_width_scale: Annotated[float, AdvancedField(default=1)] = 1
    line_width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    line_width_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    stroked: Annotated[bool, AdvancedField(default=False)] = False
    filled: Annotated[bool, AdvancedField(default=True)] = True
    elevation_scale: Annotated[float, AdvancedField(default=1)] = 1
    billboard: Annotated[bool, AdvancedField(default=False)] = False


class PolygonLayerStyle(LayerStyleBase):
    """
    Polygon Layer style kwargs
    See https://deck.gl/docs/api-reference/layers/polygon-layer for more info
    """

    get_polygon: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_fill_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_line_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_elevation: Annotated[
        FloatAccessor | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None
    get_line_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    line_width_scale: Annotated[float, AdvancedField(default=1)] = 1
    line_width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    line_width_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    line_miter_limit: Annotated[float, AdvancedField(default=4)] = 4
    line_joint_rounded: Annotated[bool, AdvancedField(default=False)] = False
    stroked: Annotated[bool, AdvancedField(default=False)] = False
    filled: Annotated[bool, AdvancedField(default=True)] = True
    billboard: Annotated[bool, AdvancedField(default=False)] = False
    antialiasing: Annotated[bool, AdvancedField(default=True)] = True
    extruded: Annotated[bool, AdvancedField(default=False)] = False
    wireframe: Annotated[bool, AdvancedField(default=True)] = True


class TextLayerStyle(LayerStyleBase):
    """
    Text Layer style kwargs
    See https://deck.gl/docs/api-reference/layers/text-layer for more info
    """

    get_position: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_text: Annotated[str, AdvancedField(default="label")] = "label"
    get_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_background_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_border_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_border_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_size: Annotated[FloatAccessor, AdvancedField(default=12)] = 12
    get_angle: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    background_border_radius: Annotated[
        float | Tuple[float, float, float, float] | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None
    background_padding: Annotated[
        Tuple[float, float] | Tuple[float, float, float, float] | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None
    billboard: Annotated[bool, AdvancedField(default=False)] = False
    background: Annotated[bool, AdvancedField(default=False)] = False
    size_scale: Annotated[float, AdvancedField(default=1)] = 1
    size_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    size_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    size_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    font_family: Annotated[PydeckString, AdvancedField(default="Monaco, monospace")] = "Monaco, monospace"
    font_weight: Annotated[PydeckString | SkipJsonSchema[None], AdvancedField(default=None)] = None
    line_height: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    outline_width: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    outline_color: Annotated[
        Tuple[float, float, float, float] | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None
    word_break: Annotated[WordBreak | SkipJsonSchema[None], AdvancedField(default=None)] = None
    max_width: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_text_anchor: Annotated[TextAnchor, AdvancedField(default="middle")] = "middle"
    get_alignment_baseline: Annotated[AlignmentBaseline, AdvancedField(default="center")] = "center"
    get_pixel_offset: Annotated[Tuple[float, float] | SkipJsonSchema[None], AdvancedField(default=None)] = None


class IconLayerStyle(LayerStyleBase):
    """
    Icon Layer style kwargs
    See https://deck.gl/docs/api-reference/layers/icon-layer for more info
    """

    get_position: Annotated[str, AdvancedField(default="geometry.coordinates")] = "geometry.coordinates"
    get_icon: Annotated[str, AdvancedField(default="icon")] = "icon"
    get_size: Annotated[int | str, AdvancedField(default=1)] = 1
    size_scale: Annotated[int, AdvancedField(default=1)] = 1
    size_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    billboard: Annotated[bool, AdvancedField(default=False)] = False


class GeoJSONLayerStyle(LayerStyleBase):
    """
    GeoJSON Layer style kwargs
    See https://deck.gl/docs/api-reference/layers/geojson-layer for more info
    """

    filled: Annotated[bool, AdvancedField(default=True)] = True
    stroked: Annotated[bool, AdvancedField(default=True)] = True
    extruded: Annotated[bool, AdvancedField(default=False)] = False
    wireframe: Annotated[bool, AdvancedField(default=False)] = False

    get_fill_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_line_color: Annotated[ColorAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_line_width: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=1)] = 1
    get_elevation: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=None)] = None
    get_point_radius: Annotated[FloatAccessor | SkipJsonSchema[None], AdvancedField(default=1)] = 1

    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    line_width_scale: Annotated[float, AdvancedField(default=1)] = 1
    line_width_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    line_width_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    line_miter_limit: Annotated[float, AdvancedField(default=4)] = 4
    line_joint_rounded: Annotated[bool, AdvancedField(default=False)] = False

    elevation_scale: Annotated[float, AdvancedField(default=1)] = 1

    point_radius_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    point_radius_scale: Annotated[float, AdvancedField(default=1)] = 1
    point_radius_min_pixels: Annotated[float, AdvancedField(default=0)] = 0
    point_radius_max_pixels: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    billboard: Annotated[bool, AdvancedField(default=False)] = False


LayerStyle = Union[
    PathLayerStyle,
    ScatterplotLayerStyle,
    PolygonLayerStyle,
    TextLayerStyle,
    IconLayerStyle,
    HexagonLayerStyle,
    GeoJSONLayerStyle,
]


class ViewState(BaseModel):
    """
    Represents a deck.gl view state
    """

    longitude: Annotated[float, AdvancedField(default=0, le=180, ge=-180)] = 0
    latitude: Annotated[float, AdvancedField(default=0, le=90, ge=-90)] = 0
    zoom: Annotated[float, AdvancedField(default=0, le=20, ge=0)] = 0
    pitch: Annotated[float, AdvancedField(default=0, le=60, ge=0)] = 0
    bearing: Annotated[float, AdvancedField(default=0, le=360)] = 0


@dataclass
class LegendValue:
    """
    An individual label/column pair to be displayed on a legend
    """

    label: LegendLabel
    color: LegendColor


@dataclass
class LegendSegment:
    """
    A legend definition by provided values
    """

    values: Annotated[list[LegendValue], Field()]
    title: Annotated[str, Field(default="Legend")] = "Legend"


@dataclass
class LegendFromDataframe:
    """
    A legend definition referencing values in a dataframe from the specified column values
    """

    title: Annotated[str, AdvancedField(default="Legend")] = "Legend"
    label_column: Annotated[str, AdvancedField(default="labels")] = "labels"
    color_column: Annotated[str, AdvancedField(default="colors")] = "colors"
    sort: Annotated[
        Literal["ascending", "descending"] | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None
    label_suffix: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None

    @property
    def display_name(self):
        return self.title.replace("_", " ").title()

    def build_legend_from_dataframe(self, df: AnyGeoDataFrame) -> LegendSegment:
        """
        Lookup the legend label/values from the provided dataframe values
        """
        lookup = df.drop_duplicates(subset=self.label_column)[[self.label_column, self.color_column]]
        if self.sort:
            lookup = lookup.sort_values(
                self.label_column,
                ascending=True if self.sort == "ascending" else False,
                # Coerce numeric strings so e.g. "10" sorts after "2" rather than before.
                key=lambda col: pd.to_numeric(col, errors="ignore"),  # type: ignore[call-overload]
            )
        return LegendSegment(
            title=self.display_name,
            values=[
                LegendValue(
                    label=f"{row[self.label_column]}{self.label_suffix}"
                    if self.label_suffix
                    else row[self.label_column],
                    color=_color_tuple_to_css(row[self.color_column]),
                )
                for _, row in lookup.iterrows()
            ],
        )


LegendDefinition = LegendFromDataframe | LegendSegment


class TiledBitmapLayerDefinition(BaseModel):
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
    opacity: Annotated[
        float,
        Field(
            default=1,
            ge=0,
            le=1,
            title="Layer Opacity",
            description="Set layer transparency from 1 (fully visible) to 0 (hidden).",
        ),
    ] = 1
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
            for layer_name in TileLayerPresets.keys():
                if layer_name == v:
                    return v
            raise ValueError(
                f"String input must match one of: {[layer_name for layer_name in TileLayerPresets.keys()]}"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def set_url(cls, values):
        if (
            isinstance(values, dict)
            and values.get("layer_name")
            and values.get("layer_name") != ""
            and values.get("layer_name") in TileLayerPresets.keys()
        ):
            values["url"] = TileLayerPresets.get(values.get("layer_name")).get("url")
        return values

    def _as_json_schema_default(self):
        default = {
            "url": self.url,
            "opacity": self.opacity,
        }
        if self.max_zoom is not None:
            default["max_zoom"] = self.max_zoom
        if self.min_zoom is not None:
            default["min_zoom"] = self.min_zoom
        return default


class BitmapLayerDefinition(BaseModel):
    """A single-image overlay positioned at geographic bounds (e.g., raster from Earth Engine)."""

    image: PydeckString
    bounds: list[float]
    opacity: float = 1.0
    legend: LegendSegment | SkipJsonSchema[None] = None


_preset_or_custom_json_schema_extra = make_preset_or_custom_json_schema_extra(
    TiledBitmapLayerDefinition, TileLayerPresets
)


@register()
def set_base_maps_pydeck(
    base_maps: Annotated[
        list[TiledBitmapLayerDefinition] | SkipJsonSchema[None],
        Field(
            json_schema_extra=_preset_or_custom_json_schema_extra,
            title=" ",
            description="Select tile layers to use as base layers in map outputs. \
            The first layer in the list will be the bottommost layer displayed.",
        ),
    ] = None,
) -> Annotated[list[TiledBitmapLayerDefinition], Field()]:
    if base_maps is None:
        base_maps = [
            TiledBitmapLayerDefinition(layer_name="TERRAIN"),
            TiledBitmapLayerDefinition(layer_name="SATELLITE", opacity=0.5),
        ]
    return base_maps


@dataclass
class PydeckLayerDefinition:
    """
    A wrapper for a defined layer
    We do this to allow passing layers around between tasks while avoiding a compile time dependency on pydeck
    """

    layer_type: str
    layer_style: LayerStyle
    legend: LegendDefinition | None
    geodataframe: AnyGeoDataFrame | None = None
    data_url: str | None = None
    zoom: bool = False

    def __post_init__(self) -> None:
        if self.geodataframe is None and self.data_url is None:
            raise ValueError("PydeckLayerDefinition requires either 'geodataframe' or 'data_url'.")
        if self.data_url is not None and self.data_url.startswith("file://"):
            raise ValueError(
                "data_url must be an http(s) URL; 'file://' URLs are not supported "
                "(use serve_local_files=True in ScreenshotConfig to serve local files via HTTP)."
            )


class LegendStyle(BaseModel):
    """
    Additional legend configuration options applied per map, rather than per layer
    """

    placement: Annotated[WidgetPlacement, AdvancedField(default="bottom-right")] = "bottom-right"
    format_title: Annotated[bool, AdvancedField(default=False)] = False

    def display_name(self, title: str) -> str:
        return title.replace("_", " ").title() if self.format_title else title


@register()
def view_state_from_geodataframes(
    geodataframes: list[AnyGeoDataFrame],
    max_zoom: float = 20,
) -> Annotated[ViewState, Field()]:
    if not geodataframes:
        return ViewState()

    # TODO consider this as a core library function
    # To avoid mypy error
    @no_type_check
    def _concat_gdfs(dfs):
        return pd.concat(dfs)

    combined = _concat_gdfs(geodataframes)

    bounds = combined.total_bounds
    bbox = [
        [bounds[0], bounds[1]],  # Northwest corner
        [bounds[2], bounds[3]],  # Southeast corner
    ]
    computed_zoom = pdk.data_utils.viewport_helpers.bbox_to_zoom_level(bbox)
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    return ViewState(longitude=center_lon, latitude=center_lat, zoom=min(max_zoom, computed_zoom))


@register()
def view_state_from_layers(
    layers: list[PydeckLayerDefinition],
    max_zoom: float = 20,
) -> Annotated[ViewState, Field()]:
    # TODO: handle view_state for 3-d layers with elevation
    gdfs = [layer.geodataframe for layer in layers if layer.geodataframe is not None]
    if not gdfs:
        logger.warning(
            "view_state_from_layers: no geodataframe data available (all layers use data_url). "
            "Falling back to default view state centred on (0, 0). "
            "Pass an explicit view_state to draw_map to override this."
        )
    return view_state_from_geodataframes(
        geodataframes=gdfs,
        max_zoom=max_zoom,
    )


def _color_tuple_to_css(color: Tuple[int, int, int, int]) -> str:
    # eg [255,0,120,255] converts to 'rgba(255,0,120,1)'
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3] / 255})"


@register()
def create_hexagon_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        HexagonLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=HexagonLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    zoom: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="If true, the map will be zoomed to the bounds of this layer",
            exclude=True,
        ),
    ] = False,
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates a hexagon layer definition based on the provided configuration.
    """
    if geodataframe is not None:
        gdf = geodataframe.to_crs("EPSG:4326")  # type: ignore[operator]
        gdf_clean = gdf[~gdf.geometry.isna()].copy()
        gdf_clean = gdf_clean[
            ~gdf_clean.geometry.apply(lambda geom: geom.is_empty) & gdf_clean.geometry.apply(lambda geom: geom.is_valid)
        ].copy()
        # Extract longitude and latitude from geometry
        gdf_clean["lng"] = gdf_clean.geometry.x
        gdf_clean["lat"] = gdf_clean.geometry.y
        geodataframe = gdf_clean

    return PydeckLayerDefinition(
        layer_type="HexagonLayer",
        layer_style=layer_style or HexagonLayerStyle(),
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
        zoom=zoom,
    )


@register()
def create_path_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        PathLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=PathLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    zoom: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="If true, the map will be zoomed to the bounds of this layer",
            exclude=True,
        ),
    ] = False,
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates a polyline layer definition based on the provided configuration.
    """
    return PydeckLayerDefinition(
        layer_type="PathLayer",
        layer_style=layer_style or PathLayerStyle(),
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
        zoom=zoom,
    )


@register()
def create_polygon_layer_pydeck(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        PolygonLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=PolygonLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    zoom: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="If true, the map will be zoomed to the bounds of this layer",
            exclude=True,
        ),
    ] = False,
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates a polyline layer definition based on the provided configuration.
    """
    return PydeckLayerDefinition(
        layer_type="PolygonLayer",
        layer_style=layer_style or PolygonLayerStyle(),
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
        zoom=zoom,
    )


@register()
def create_scatterplot_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        ScatterplotLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=ScatterplotLayerStyle(),
            description="Style arguments for the layer.",
        ),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    zoom: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="If true, the map will be zoomed to the bounds of this layer",
            exclude=True,
        ),
    ] = False,
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates a scatterplot layer definition based on the provided configuration.
    """
    layer_style = layer_style if layer_style else ScatterplotLayerStyle()

    if isinstance(layer_style.get_radius, str):
        radius_series = geodataframe[layer_style.get_radius]

        # Lift all values up such that the min == 1
        if radius_series.min() < 0:
            radius_series = radius_series + (0 - radius_series.min()) + 1

        # set to 0, and lift everything else by 1 to distinguish NaN's and minimums
        if radius_series.hasnans:
            radius_series = radius_series + 1
            radius_series = radius_series.fillna(1)

        layer_style.get_radius = radius_series.values  # type: ignore[assignment]

    return PydeckLayerDefinition(
        layer_type="ScatterplotLayer",
        layer_style=layer_style,
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
        zoom=zoom,
    )


@register()
def create_text_layer_pydeck(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        TextLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=TextLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    zoom: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="If true, the map will be zoomed to the bounds of this layer",
            exclude=True,
        ),
    ] = False,
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates a text layer definition based on the provided configuration.
    """
    return PydeckLayerDefinition(
        layer_type="TextLayer",
        layer_style=layer_style or TextLayerStyle(),
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
        zoom=zoom,
    )


@register()
def create_icon_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        IconLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=IconLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    zoom: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="If true, the map will be zoomed to the bounds of this layer",
            exclude=True,
        ),
    ] = False,
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates an icon layer definition based on the provided configuration.
    """
    return PydeckLayerDefinition(
        layer_type="IconLayer",
        layer_style=layer_style or IconLayerStyle(),
        # TODO support icons in legend
        legend=None,
        geodataframe=geodataframe,
        data_url=data_url,
        zoom=zoom,
    )


@register()
def create_geojson_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame | SkipJsonSchema[None],
        Field(description="The geodataframe to visualize.", exclude=True),
    ] = None,
    data_url: Annotated[
        str | SkipJsonSchema[None],
        Field(description="URL to a GeoJSON file to visualize."),
    ] = None,
    layer_style: Annotated[
        GeoJSONLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=GeoJSONLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    zoom: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="If true, the map will be zoomed to the bounds of this layer",
            exclude=True,
        ),
    ] = False,
) -> Annotated[PydeckLayerDefinition, Field()]:
    """
    Creates a GeoJSON layer definition based on the provided configuration.
    """
    return PydeckLayerDefinition(
        layer_type="GeoJsonLayer",
        layer_style=layer_style or GeoJSONLayerStyle(),
        legend=legend,
        geodataframe=geodataframe,
        data_url=data_url,
        zoom=zoom,
    )


@register()
def draw_map(
    geo_layers: Annotated[
        PydeckLayerDefinition | list[PydeckLayerDefinition] | SkipJsonSchema[None],
        Field(description="A list of map layers to add to the map.", exclude=True),
    ] = None,
    tile_layers: Annotated[
        list[TiledBitmapLayerDefinition | BitmapLayerDefinition] | SkipJsonSchema[None],
        Field(description="A list of tile layers (base maps and/or overlays)."),
    ] = None,
    static: Annotated[bool, Field(description="Set to true to disable map pan/zoom.")] = False,
    title: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default="",
            description="""\
            The map title. Note this is the title drawn on the map canvas itself, and will result
            in duplicate titles if set in the context of a dashboard in which the iframe/widget
            container also has a title set on it.
            """,
        ),
    ] = None,
    legend_style: Annotated[
        LegendStyle | SkipJsonSchema[None],
        AdvancedField(
            default=LegendStyle(),
            description="Additional arguments for configuring the legend.",
        ),
    ] = None,
    max_zoom: Annotated[
        int,
        AdvancedField(
            default=20,
            description="""\
            The maximum zoom level allowed by the map.
            This setting will be overridden if provided
            tile layers max zoom levels are lower than this value.
            """,
        ),
    ] = 20,
    view_state: Annotated[
        ViewState | SkipJsonSchema[None],
        AdvancedField(
            default=ViewState(),
            description="Manually set the view state of the map.",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this tile layer belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    """
    Creates a map based on the provided layer definitions and configuration.

    Args:
    geo_layers (PydeckLayerDefinition | list[PydeckLayerDefinition] | None): Map layers to add to the map.
    tile_layers (list): A named tile layer, ie OpenStreetMap.
    static (bool): Set to true to disable map pan/zoom.
    title (str): The map title.
    legend_style (WidgetStyleBase): Additional arguments for configuring the Legend.
    max_zoom (int): The maximum zoom level of the map
    view_state (ViewState): Manually set the view state of the map, overrides any layer zoom settings.
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    str: A static HTML representation of the map.
    """
    pdk.settings.custom_libraries = PYDECK_CUSTOM_LIBRARIES

    DEFAULT_WIDGETS = [
        # TODO ids can be removed once upstream pydeck changes are released
        pdk.Widget(
            "NorthArrowWidget",
            placement="top-left",
            id="NorthArrowWidget",
            style={"transform": "scale(0.8)"},
        ),
        pdk.Widget("ScaleWidget", placement="bottom-left", id="ScaleWidget"),
        pdk.Widget("SaveImageWidget", placement="top-right", id="SaveImageWidget"),
    ]

    if tile_layers is None:
        tile_layers = []
    else:
        tile_layers = list(tile_layers)
    if legend_style is None:
        legend_style = LegendStyle()

    legend_values: list = []
    map_layers: list = []
    map_widgets: list = DEFAULT_WIDGETS.copy()

    for tile_layer in tile_layers:
        if isinstance(tile_layer, BitmapLayerDefinition):
            dump = tile_layer.model_dump(exclude_none=True)
            dump.pop("legend", None)
            layer = pdk.Layer("BitmapLayer", **dump)
            map_layers.append(layer)
            if tile_layer.legend is not None:
                legend_values.append(tile_layer.legend)
        else:
            layer = pdk.Layer(
                "TiledBitmapLayer",
                data=tile_layer.url,
                max_zoom=tile_layer.max_zoom,
                min_zoom=tile_layer.min_zoom,
                opacity=tile_layer.opacity,
                tile_size=256,
                widget_id=pdk.types.String(widget_id),
            )
            map_layers.append(layer)
            if tile_layer.max_zoom < max_zoom:
                max_zoom = tile_layer.max_zoom

    # Normalize geo_layers to a list
    if geo_layers is None:
        geo_layers = []
    elif isinstance(geo_layers, PydeckLayerDefinition):
        geo_layers = [geo_layers]
    for layer_def in geo_layers:
        # Rendering: prefer data_url if set, fall back to geodataframe
        if layer_def.data_url is not None:
            data = pdk.types.String(layer_def.data_url)
        elif layer_def.geodataframe is not None:
            gdf = layer_def.geodataframe.to_crs("EPSG:4326")  # type: ignore[operator]
            # Pydeck's PolygonLayer does not support MultiPolygon geometries,
            # so we explode them into individual Polygons.
            is_multi = gdf.geometry.geom_type == "MultiPolygon"
            if is_multi.any():
                gdf = pd.concat(
                    [gdf[~is_multi], gdf[is_multi].explode(index_parts=False)],
                    ignore_index=True,
                )
            data = gdf

        layer = pdk.Layer(
            type=layer_def.layer_type,
            data=data,
            **layer_def.layer_style.model_dump(exclude_none=True),
        )
        map_layers.append(layer)

        # Legend: use geodataframe if present (regardless of rendering path)
        if legend_def := layer_def.legend:
            if isinstance(legend_def, LegendSegment):
                legend_values.append(legend_def)
            elif isinstance(legend_def, LegendFromDataframe):
                if layer_def.geodataframe is not None:
                    legend_values.append(legend_def.build_legend_from_dataframe(layer_def.geodataframe))
                else:
                    logger.warning(
                        "LegendFromDataframe legend skipped for layer '%s': "
                        "no geodataframe is available (layer uses data_url). "
                        "Use a LegendSegment to define a static legend for URL-backed layers.",
                        layer_def.layer_type,
                    )

    if legend_values:
        if legend_style.format_title:
            legend_values = [
                LegendSegment(title=legend_style.display_name(lv.title), values=lv.values) for lv in legend_values
            ]
        map_widgets.append(
            pdk.Widget(
                "LegendWidget",
                id="LegendWidget",  # TODO remove this once upstream pydeck changes are released
                legend_values=legend_values,
                placement=legend_style.placement,
            )
        )

    if title:
        map_widgets.append(
            pdk.Widget(
                "TitleWidget",
                id="TitleWidget",  # TODO remove this once upstream pydeck changes are released
                title=title,
            )
        )

    if view_state is None:
        zoom_layers = [layer for layer in geo_layers if layer.zoom]
        if not zoom_layers:
            zoom_layers = geo_layers
        view_state = view_state_from_layers(layers=zoom_layers, max_zoom=max_zoom)

    m = pdk.Deck(
        layers=map_layers,
        widgets=map_widgets,
        initial_view_state=view_state,
        # The only non-default value here is repeat=True
        # which in our case allows tile layers to repeat/wrap at high zoom levels
        views=pdk.View(
            "MapView",
            controller=not static,
            repeat=True,
        ),
        # In order to avoid issues with z-fighting,
        # explicitly disable depth testing when layers have no extrusions
        parameters={"depthTest": any([getattr(layer, "extruded", False) for layer in map_layers])},
        map_style=pdk.map_styles.LIGHT_NO_LABELS,
    )

    html = m.to_html(as_string=True)
    return html


@register()
def rewrite_file_urls_for_screenshots(
    html: Annotated[str, Field(description="HTML string output from draw_map.")],
    file_urls: Annotated[list[str], Field(description="The file url strings to replace in `html`.")],
) -> Annotated[str, Field()]:
    """
    Rewrites file_urls in map HTML to ``http://127.0.0.1:<port>/``
    so that Playwright can fetch local files without CORS restrictions when
    ``serve_local_files=True`` is set on ``ScreenshotConfig``.

    Only the filename (stem + extension) is preserved — the full local path is
    dropped. For example ``file:///some/long/path/data.geojson`` becomes
    ``http://127.0.0.1:8099/data.geojson``.

    The port defaults to ``8099`` and can be overridden via the
    ``ECOSCOPE_SCREENSHOT_FILE_SERVER_PORT`` environment variable.
    """
    import os

    port = int(os.environ.get("ECOSCOPE_SCREENSHOT_FILE_SERVER_PORT", 8099))
    base_url = f"http://127.0.0.1:{port}"

    for file_url in file_urls:
        filename = os.path.basename(file_url)
        html = html.replace(file_url, f"{base_url}/{filename}")

    return html


@register()
def create_tiled_bitmap_layer(
    url: Annotated[
        str,
        Field(
            description="The tile URL template with {z}, {x}, {y} placeholders.",
        ),
    ],
    opacity: Annotated[
        float,
        Field(
            description="Layer opacity from 0 to 1.",
            ge=0,
            le=1,
        ),
    ] = 1.0,
    max_zoom: Annotated[
        int,
        AdvancedField(
            description="Maximum zoom level.",
            default=20,
        ),
    ] = 20,
    min_zoom: Annotated[
        int,
        AdvancedField(
            description="Minimum zoom level.",
            default=0,
        ),
    ] = 0,
) -> Annotated[TiledBitmapLayerDefinition, Field()]:
    """Creates a tiled bitmap layer definition from a tile URL."""
    return TiledBitmapLayerDefinition(
        url=url,
        opacity=opacity,
        max_zoom=max_zoom,
        min_zoom=min_zoom,
    )


@register()
def merge_tile_layers(
    base_layers: Annotated[
        list[TiledBitmapLayerDefinition] | SkipJsonSchema[None],
        Field(description="Static base tile layers to prepend."),
    ] = None,
    overlay: Annotated[
        BitmapLayerDefinition | SkipJsonSchema[None],
        Field(description="Per-group overlay tile layer to append."),
    ] = None,
) -> Annotated[list[TiledBitmapLayerDefinition | BitmapLayerDefinition], Field()]:
    """Merges static base tile layers with a per-group overlay into a single list."""
    layers: list[TiledBitmapLayerDefinition | BitmapLayerDefinition] = []
    if base_layers:
        layers.extend(base_layers)
    if overlay:
        layers.append(overlay)
    return layers
