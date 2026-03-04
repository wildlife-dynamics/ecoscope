import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

UnitType = Literal["meters", "pixels", "common"]
WidgetPlacement = Literal[
    "top-left", "top-right", "bottom-left", "bottom-right", "fill"
]

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


class LayerStyleBase(BaseModel):
    auto_highlight: Annotated[bool, AdvancedField(default=False)] = False
    opacity: OpacityAnnotation = 1
    pickable: Annotated[bool, AdvancedField(default=True)] = True


class TextLayerStyle(LayerStyleBase):
    get_text: Annotated[str, AdvancedField(default="label")] = "label"
    get_color: (
        str
        | SkipJsonSchema[list[int]]
        | SkipJsonSchema[list[list[int]]]
        | SkipJsonSchema[None]
    ) = None
    font_family: Annotated[str, AdvancedField(default="Monaco, monospace")] = (
        "Monaco, monospace"
    )
    font_weight: Annotated[str, AdvancedField(default="bold")] = "normal"
    get_size: Annotated[float, AdvancedField(default=12)] = 12
    get_text_anchor: Annotated[str, AdvancedField(default="middle")] = "middle"
    get_alignment_baseline: Annotated[str, AdvancedField(default="center")] = "center"
    get_background_color: (
        str
        | SkipJsonSchema[list[int]]
        | SkipJsonSchema[list[list[int]]]
        | SkipJsonSchema[None]
    ) = None


class PolylineLayerStyle(LayerStyleBase):
    get_color: (
        str
        | SkipJsonSchema[list[int]]
        | SkipJsonSchema[list[list[int]]]
        | SkipJsonSchema[None]
    ) = None
    get_width: Annotated[float, AdvancedField(default=3)] = 3
    color_column: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = (
        None
    )
    width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    cap_rounded: Annotated[bool, AdvancedField(default=True)] = True


class ShapeLayerStyle(LayerStyleBase):
    filled: Annotated[bool, AdvancedField(default=True)] = True
    get_fill_color: (
        str
        | SkipJsonSchema[list[int]]
        | SkipJsonSchema[list[list[int]]]
        | SkipJsonSchema[None]
    ) = None
    get_line_color: (
        str
        | SkipJsonSchema[list[int]]
        | SkipJsonSchema[list[list[int]]]
        | SkipJsonSchema[None]
    ) = None
    get_line_width: Annotated[float, AdvancedField(default=1)] = 1
    fill_color_column: Annotated[
        str | SkipJsonSchema[None], AdvancedField(default=None)
    ] = None
    line_width_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    stroked: Annotated[bool, AdvancedField(default=False)] = False


class PointLayerStyle(ShapeLayerStyle):
    get_radius: Annotated[
        float
        | str
        | SkipJsonSchema[list[int]]
        | SkipJsonSchema[list[list[int]]]
        | SkipJsonSchema[None],
        AdvancedField(default=5),
    ] = 5
    radius_units: Annotated[UnitType, AdvancedField(default="pixels")] = "pixels"
    radius_scale: Annotated[float, AdvancedField(default=1)] = 1


class PolygonLayerStyle(ShapeLayerStyle):
    extruded: Annotated[bool, AdvancedField(default=False)] = False
    get_elevation: Annotated[float, AdvancedField(default=1000)] = 1000


LayerStyle = Union[
    PolylineLayerStyle, PointLayerStyle, PolygonLayerStyle, TextLayerStyle
]


class NorthArrowStyle(BaseModel):
    placement: Annotated[WidgetPlacement, AdvancedField(default="top-left")] = (
        "top-left"
    )
    style: Annotated[dict, AdvancedField(default={"transform": "scale(0.8)"})] = {
        "transform": "scale(0.8)"
    }


class LegendStyle(BaseModel):
    placement: Annotated[WidgetPlacement, AdvancedField(default="bottom-right")] = (
        "bottom-right"
    )
    title: Annotated[str, AdvancedField(default="Legend")] = "Legend"
    format_title: Annotated[bool, AdvancedField(default=False)] = False

    @property
    def display_name(self):
        return self.title.replace("_", " ").title() if self.format_title else self.title


class ViewState(BaseModel):
    longitude: Annotated[float, AdvancedField(default=0, le=180, ge=-180)] = 0
    latitude: Annotated[float, AdvancedField(default=0, le=90, ge=-90)] = 0
    zoom: Annotated[float, AdvancedField(default=0, le=20, ge=0)] = 0
    pitch: Annotated[float, AdvancedField(default=0, le=60, ge=0)] = 0
    bearing: Annotated[float, AdvancedField(default=0, le=360, ge=0)] = 0


@dataclass
class LegendDefinition:
    label_column: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = (
        None
    )
    color_column: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = (
        None
    )
    labels: list[str] | SkipJsonSchema[None] = None
    colors: list[str] | SkipJsonSchema[None] = None
    sort: Annotated[
        Literal["ascending", "descending"] | None, AdvancedField(default=None)
    ] = None
    label_suffix: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = (
        None
    )


@dataclass
class LayerDefinition:
    geodataframe: AnyGeoDataFrame
    layer_style: LayerStyle
    legend: LegendDefinition | None
    tooltip_columns: list[str] | None
    zoom: bool = False


TileLayerPresets = {
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
    "LANDDX": {
        "url": "https://tiles.arcgis.com/tiles/POUcpLYXNckpLjnY/arcgis/rest/services/landDx_basemap_tiles_mapservice/MapServer/tile/{z}/{y}/{x}",
        "title": "LandDx",
    },
    "USGS HILLSHADE": {
        "url": "https://server.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        "title": "USGS Hillshade",
    },
}


class TileLayer(BaseModel):
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
        int | SkipJsonSchema[None],
        Field(
            default=None,
            title="Layer Max Zoom",
            description="Set the maximum zoom level to fetch tiles for.",
        ),
    ] = None
    min_zoom: Annotated[
        int | SkipJsonSchema[None],
        Field(
            default=None,
            title="Layer Min Zoom",
            description="Set the minimum zoom level to fetch tiles for.",
        ),
    ] = None

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
        if self.max_zoom:
            default["max_zoom"] = self.max_zoom
        if self.min_zoom:
            default["min_zoom"] = self.min_zoom
        return default


def _custom_tile_layer_json_schema() -> dict:
    schema = TileLayer.model_json_schema()
    schema["properties"]["url"]["title"] = "Custom Layer URL"
    schema["properties"]["opacity"]["title"] = "Custom Layer Opacity"
    schema["properties"]["max_zoom"]["title"] = "Custom Layer Max Zoom"
    schema["properties"]["min_zoom"]["title"] = "Custom Layer Min Zoom"
    schema["title"] = "Custom Layer (Advanced)"
    return schema


def _preset_tile_layer_json_schema(preset_name: str) -> dict:
    schema = TileLayer.model_json_schema()
    url = TileLayerPresets.get(preset_name, {}).get("url")
    title = TileLayerPresets.get(preset_name, {}).get("title")
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


def _preset_or_custom_json_schema_extra(schema: dict) -> None:
    schema["items"]["title"] = "Base Layer"
    schema["items"]["anyOf"] = [
        _preset_tile_layer_json_schema(preset) for preset in TileLayerPresets.keys()
    ]
    schema["items"]["anyOf"].append(_custom_tile_layer_json_schema())
    schema["default"] = [
        TileLayer(layer_name="TERRAIN")._as_json_schema_default(),
        TileLayer(layer_name="SATELLITE", opacity=0.5)._as_json_schema_default(),
    ]
    schema["ecoscope:advanced"] = True
    schema["items"].pop("$ref")


@register()
def set_layer_opacity(
    opacity: OpacityAnnotation = 1,
) -> float:
    return opacity


@register()
def set_base_maps(
    base_maps: Annotated[
        list[TileLayer] | SkipJsonSchema[None],
        Field(
            json_schema_extra=_preset_or_custom_json_schema_extra,
            title=" ",
            description="Select tile layers to use as base layers in map outputs. The first layer in the list will be the bottommost layer displayed.",
        ),
    ] = None,
) -> Annotated[list[TileLayer], Field()]:
    if base_maps is None:
        base_maps = [
            TileLayer(layer_name="TERRAIN"),
            TileLayer(layer_name="SATELLITE", opacity=0.5),
        ]
    return base_maps


@register()
def create_polyline_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="The geodataframe to visualize.", exclude=True),
    ],
    layer_style: Annotated[
        PolylineLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=PolylineLayerStyle(), description="Style arguments for the layer."
        ),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    tooltip_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, only the listed dataframe columns will display in the layer's picking info",
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
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates a polyline layer definition based on the provided configuration.

    Args:
    geodataframe (geopandas.GeoDataFrame): The geodataframe to visualize.
    layer_style (PolylineLayerStyle): Style arguments for the data visualization.

    Returns:
    The generated LayerDefinition
    """

    return LayerDefinition(
        geodataframe=geodataframe,
        layer_style=layer_style if layer_style else PolylineLayerStyle(),
        legend=legend,
        tooltip_columns=tooltip_columns,
        zoom=zoom,
    )


@register()
def create_polygon_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="The geodataframe to visualize.", exclude=True),
    ],
    layer_style: Annotated[
        PolygonLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=PolygonLayerStyle(), description="Style arguments for the layer."
        ),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    tooltip_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, only the listed dataframe columns will display in the layer's picking info",
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
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates a polygon layer definition based on the provided configuration.

    Args:
    geodataframe (geopandas.GeoDataFrame): The geodataframe to visualize.
    layer_style (PolygonLayerStyle): Style arguments for the data visualization.

    Returns:
    The generated LayerDefinition
    """

    return LayerDefinition(
        geodataframe=geodataframe,
        layer_style=layer_style if layer_style else PolygonLayerStyle(),
        legend=legend,  # type: ignore[arg-type]
        tooltip_columns=tooltip_columns,
        zoom=zoom,
    )


@register()
def create_point_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="The geodataframe to visualize.", exclude=True),
    ],
    layer_style: Annotated[
        PointLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=PointLayerStyle(), description="Style arguments for the layer."
        ),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    tooltip_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, only the listed dataframe columns will display in the layer's picking info",
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
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates a point layer definition based on the provided configuration.

    Args:
    geodataframe (geopandas.GeoDataFrame): The geodataframe to visualize.
    layer_style (LayerStyle): Style arguments for the data visualization.

    Returns:
    The generated LayerDefinition
    """
    layer_style = layer_style if layer_style else PointLayerStyle()

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

    return LayerDefinition(
        geodataframe=geodataframe,
        layer_style=layer_style,
        legend=legend,  # type: ignore[arg-type]
        tooltip_columns=tooltip_columns,
        zoom=zoom,
    )


@register()
def create_text_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="The geodataframe to visualize.", exclude=True),
    ],
    layer_style: Annotated[
        TextLayerStyle | SkipJsonSchema[None],
        AdvancedField(
            default=TextLayerStyle(), description="Style arguments for the layer."
        ),
    ] = None,
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates a text layer definition based on the provided configuration.

    Args:
    geodataframe (geopandas.GeoDataFrame): The geodataframe to visualize.
    layer_style (LayerStyle): Style arguments for the data visualization.

    Returns:
    The generated LayerDefinition
    """

    return LayerDefinition(
        geodataframe=geodataframe,
        layer_style=layer_style if layer_style else TextLayerStyle(),
        legend=None,
        tooltip_columns=None,
    )


@register()
def draw_ecomap(
    geo_layers: Annotated[
        LayerDefinition | list[LayerDefinition],
        Field(description="A list of map layers to add to the map.", exclude=True),
    ],
    tile_layers: Annotated[
        list[TileLayer] | SkipJsonSchema[None],
        Field(description="A list of named tile layer with opacity, ie OpenStreetMap."),
    ] = None,
    static: Annotated[
        bool, Field(description="Set to true to disable map pan/zoom.")
    ] = False,
    title: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="""\
            The map title. Note this is the title drawn on the map canvas itself, and will result
            in duplicate titles if set in the context of a dashboard in which the iframe/widget
            container also has a title set on it.
            """,
        ),
    ] = None,
    north_arrow_style: Annotated[
        NorthArrowStyle | SkipJsonSchema[None],
        Field(description="Additional arguments for configuring the North Arrow."),
    ] = None,
    legend_style: Annotated[
        LegendStyle | SkipJsonSchema[None],
        Field(description="Additional arguments for configuring the legend."),
    ] = None,
    max_zoom: Annotated[
        int,
        Field(description="Max zoom level."),
    ] = 20,
    view_state: Annotated[
        ViewState | SkipJsonSchema[None],
        Field(
            description="Manually set the view state of the map, overrides any layer zoom settings.",
            exclude=True,
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
    geo_layers (LayerDefinition | list[LayerDefinition]): A list of map layers to add to the map.
    tile_layers (list): A named tile layer, ie OpenStreetMap.
    static (bool): Set to true to disable map pan/zoom.
    title (str): The map title.
    north_arrow_style (NorthArrowStyle): Additional arguments for configuring the North Arrow.
    legend_style (WidgetStyleBase): Additional arguments for configuring the Legend.
    max_zoom (int): The maximum zoom level of the map
    view_state (ViewState): Manually set the view state of the map, overrides any layer zoom settings.
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    str: A static HTML representation of the map.
    """
    import pandas as pd  # type: ignore[import-untyped]
    from ecoscope.mapping import EcoMap  # type: ignore[import-untyped]
    from lonboard import BitmapTileLayer  # type: ignore[import-untyped]
    from lonboard.experimental import TextLayer  # type: ignore[import-untyped]

    if tile_layers is None:
        tile_layers = []
    if north_arrow_style is None:
        north_arrow_style = NorthArrowStyle()
    if legend_style is None:
        legend_style = LegendStyle()

    legend_labels: list = []
    legend_colors: list = []
    zoom_layers: list = []

    m = EcoMap(static=static, default_widgets=False)

    if title:
        m.add_title(title)

    m.add_scale_bar()
    m.add_north_arrow(**(north_arrow_style.model_dump(exclude_none=True)))  # type: ignore[union-attr]
    m.add_save_image()

    for tile_layer in tile_layers:
        layer = BitmapTileLayer(
            data=tile_layer.url,  # type: ignore[arg-type]
            max_zoom=tile_layer.max_zoom,  # type: ignore[arg-type]
            min_zoom=tile_layer.min_zoom,  # type: ignore[arg-type]
            opacity=tile_layer.opacity,  # type: ignore[arg-type]
            tile_size=256,  # type: ignore[arg-type]
            widget_id=widget_id,  # type: ignore[arg-type]
        )
        m.add_layer(layer)

    geo_layers = [geo_layers] if not isinstance(geo_layers, list) else geo_layers
    for layer_def in geo_layers:
        match layer_def.layer_style:
            case PointLayerStyle():
                layer = EcoMap.point_layer(
                    layer_def.geodataframe,
                    tooltip_columns=layer_def.tooltip_columns,
                    **layer_def.layer_style.model_dump(exclude_none=True),
                )
            case PolylineLayerStyle():
                layer = EcoMap.polyline_layer(
                    layer_def.geodataframe,
                    tooltip_columns=layer_def.tooltip_columns,
                    **layer_def.layer_style.model_dump(exclude_none=True),
                )
            case PolygonLayerStyle():
                layer = EcoMap.polygon_layer(
                    layer_def.geodataframe,
                    tooltip_columns=layer_def.tooltip_columns,
                    **layer_def.layer_style.model_dump(exclude_none=True),
                )
            case TextLayerStyle():
                ls = layer_def.layer_style
                layer = TextLayer.from_geopandas(
                    layer_def.geodataframe,  # type: ignore[call-arg]
                    get_text=layer_def.geodataframe.label,  # type: ignore[call-arg]
                    get_color=ls.get_color,  # type: ignore[call-arg]
                    font_family=ls.font_family,  # type: ignore[call-arg]
                    font_weight=ls.font_weight,  # type: ignore[call-arg]
                    get_size=ls.get_size,  # type: ignore[call-arg]
                    get_text_anchor=ls.get_text_anchor,  # type: ignore[call-arg]
                    get_alignment_baseline=ls.get_alignment_baseline,  # type: ignore[call-arg]
                    get_background_color=ls.get_background_color,  # type: ignore[call-arg]
                    pickable=ls.pickable,  # type: ignore[call-arg]
                )

        if layer_def.legend:
            if layer_def.legend.label_column and layer_def.legend.color_column:
                lookup = layer_def.geodataframe.drop_duplicates(
                    subset=layer_def.legend.color_column
                )
                if layer_def.legend.sort:
                    lookup = lookup.sort_values(
                        layer_def.legend.label_column,
                        ascending=True
                        if layer_def.legend.sort == "ascending"
                        else False,
                        # Attempt to coerce numeric strings to numbers
                        # in order to maintain a proper numeric sort
                        key=lambda col: pd.to_numeric(col, errors="ignore"),  # type: ignore[call-overload]
                    )
                for _, row in lookup.iterrows():
                    if row[layer_def.legend.color_column] not in legend_colors:
                        legend_labels.append(row[layer_def.legend.label_column])
                        legend_colors.append(row[layer_def.legend.color_column])
            elif layer_def.legend.labels and layer_def.legend.colors:
                legend_labels.extend(layer_def.legend.labels)
                legend_colors.extend(layer_def.legend.colors)
            if legend_labels and layer_def.legend.label_suffix:
                legend_labels = [
                    label + layer_def.legend.label_suffix for label in legend_labels
                ]

        if layer_def.zoom:
            zoom_layers.append(layer)

        m.add_layer(layer)

    if len(legend_labels) > 0:
        m.add_legend(
            labels=[str(ll) for ll in legend_labels],
            colors=legend_colors,
            title=legend_style.display_name,
            placement=legend_style.placement,
        )

    if view_state is None:
        if not zoom_layers:
            zoom_layers = m.layers
        m.zoom_to_bounds(feat=zoom_layers, max_zoom=max_zoom)
    else:
        m.set_view_state(**view_state.model_dump())

    return m.to_html()
