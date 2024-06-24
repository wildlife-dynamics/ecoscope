import ee
import base64
import rasterio
import json
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from typing import Dict, List, Union
from ecoscope.analysis.speed import SpeedDataFrame
from lonboard import Map
from lonboard._geoarrow.ops.bbox import Bbox
from lonboard._viewport import compute_view, bbox_to_zoom_level
from lonboard._layer import BaseLayer, BitmapLayer, BitmapTileLayer
from lonboard._viz import viz_layer
from lonboard.colormap import apply_categorical_cmap
from lonboard._deck_widget import (
    BaseDeckWidget,
    NorthArrowWidget,
    ScaleWidget,
    LegendWidget,
    TitleWidget,
    SaveImageWidget,
    FullscreenWidget,
)


class EcoMapMixin:
    def add_speedmap(
        self,
        trajectory: gpd.GeoDataFrame,
        classification_method: str = "equal_interval",
        num_classes: int = 6,
        speed_colors: List = None,
        bins: List = None,
        legend: bool = True,
    ):

        speed_df = SpeedDataFrame.from_trajectory(
            trajectory=trajectory,
            classification_method=classification_method,
            num_classes=num_classes,
            speed_colors=speed_colors,
            bins=bins,
        )

        colors = speed_df["speed_colour"].to_list()
        rgb = []
        for i, color in enumerate(colors):
            color = color.strip("#")
            rgb.append(list(int(color[i : i + 2], 16) for i in (0, 2, 4)))

        cmap = apply_categorical_cmap(values=speed_df.index.to_series(), cmap=rgb)
        path_kwargs = {"get_color": cmap, "pickable": False}
        self.add_gdf(speed_df, path_kwargs=path_kwargs)

        if legend:
            self.add_legend(labels=speed_df.label.to_list(), colors=speed_df.speed_colour.to_list())

        return speed_df


class EcoMap2(EcoMapMixin, Map):
    def __init__(self, static=False, default_widgets=True, *args, **kwargs):

        kwargs["height"] = kwargs.get("height", 600)
        kwargs["width"] = kwargs.get("width", 800)

        kwargs["layers"] = kwargs.get("layers", [self.get_named_tile_layer("OpenStreetMap")])

        if static:
            kwargs["controller"] = False

        if kwargs.get("deck_widgets") is None and default_widgets:
            kwargs["deck_widgets"] = [FullscreenWidget(), ScaleWidget(), SaveImageWidget()]

        super().__init__(*args, **kwargs)

    def add_layer(self, layer: Union[BaseLayer, List[BaseLayer]], zoom: bool = False):
        """
        Adds a layer or list of layers to the map
        Parameters
        ----------
        layer : lonboard.BaseLayer or list[lonboard.BaseLayer]
        zoom: bool
            Whether to zoom the map to the new layer
        """
        update = self.layers.copy()
        if not isinstance(layer, list):
            layer = [layer]
        update.extend(layer)
        self.layers = update
        if zoom:
            self.zoom_to_bounds(layer)

    def add_widget(self, widget: BaseDeckWidget):
        """
        Adds a deck widget to the map
        Parameters
        ----------
        widget : lonboard.BaseDeckWidget or list[lonboard.BaseDeckWidget]
        """
        update = self.deck_widgets.copy()
        update.append(widget)
        self.deck_widgets = update

    def add_gdf(self, data: Union[gpd.GeoDataFrame, gpd.GeoSeries], zoom: bool = True, **kwargs):
        """
        Visualize a gdf on the map, results in one or more layers being added
        Parameters
        ----------
        data : lonboard.BaseDeckWidget or list[lonboard.BaseDeckWidget]
        zoom : bool
            Whether or not to zoom the map to the bounds of the data
        kwargs:
            Additional kwargs passed to lonboard.viz_layer
        """
        data = data.copy()
        data = data.to_crs(4326)
        data = data.loc[(~data.geometry.isna()) & (~data.geometry.is_empty)]

        if isinstance(data, gpd.GeoDataFrame):
            for col in data:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    data[col] = data[col].astype("string")

        self.add_layer(viz_layer(data=data, **kwargs))

        if zoom:
            self.zoom_to_bounds(data)

    def add_legend(self, **kwargs):
        """
        Adds a legend to the map
        Parameters
        ----------
        placement: str
            One of "top-left", "top-right", "bottom-left", "bottom-right" or "fill"
            Where to place the widget within the map
        title: str
            A title displayed on the widget
        labels: list[str]
            A list of labels
        colors: list[str]
            A list of colors as hex values
        style: dict
            Additional style params
        """
        self.add_widget(LegendWidget(**kwargs))

    def add_north_arrow(self, **kwargs):
        """
        Adds a north arrow to the map
        Parameters
        ----------
        placement: str, one of "top-left", "top-right", "bottom-left", "bottom-right" or "fill"
            Where to place the widget within the map
        style: dict
            Additional style params
        """
        self.add_widget(NorthArrowWidget(**kwargs))

    def add_scale_bar(self, **kwargs):
        """
        Adds a scale bar to the map
        Parameters
        ----------
        placement: str, one of "top-left", "top-right", "bottom-left", "bottom-right" or "fill"
            Where to place the widget within the map
        use_imperial: bool
            If true, show scale in miles/ft, rather than m/km
        style: dict
            Additional style params
        """
        self.add_widget(ScaleWidget(**kwargs))

    def add_title(self, **kwargs):
        """
        Adds a title to the map
        Parameters
        ----------
        placement: str, one of "top-left", "top-right", "bottom-left", "bottom-right" or "fill"
            Where to place the widget within the map
        title: str
            The map title
        style: dict
            Additional style params
        """
        kwargs["placement"] = kwargs.get("placement", "fill")
        kwargs["style"] = kwargs.get("style", {"position": "relative", "margin": "0 auto", "width": "35%"})

        self.add_widget(TitleWidget(**kwargs))

    def add_save_image(self, **kwargs):
        """
        Adds a button to save the map as a png
        Parameters
        ----------
        placement: str, one of "top-left", "top-right", "bottom-left", "bottom-right" or "fill"
            Where to place the widget within the map
        style: dict
            Additional style params
        """
        self.add_widget(SaveImageWidget(**kwargs))

    def add_ee_layer(
        self,
        ee_object: Union[ee.Image, ee.ImageCollection, ee.Geometry, ee.FeatureCollection],
        visualization_params: Dict,
        **kwargs
    ):
        """
        Adds a provided Earth Engine object to the map.
        If an EE.Image/EE.ImageCollection or EE.FeatureCollection is provided,
        this results in a BitmapTileLayer being added

        For EE.Geometry objects, a list of ScatterplotLayer,PathLayer and PolygonLayer will be added
        based on the geometry itself (defers to lonboard.viz)

        Parameters
        ----------
        ee_object: ee.Image, ee.ImageCollection, ee.Geometry, ee.FeatureCollection]
            The ee object to represent as a layer
        visualization_params: dict
            Visualization params passed to EarthEngine
        kwargs
            Additional params passed to either lonboard.BitmapTileLayer or lonboard.viz

        Returns
        -------
        None
        """
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(visualization_params)
            ee_layer = BitmapTileLayer(data=map_id_dict["tile_fetcher"].url_format, **kwargs)

        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
            ee_layer = BitmapTileLayer(data=map_id_dict["tile_fetcher"].url_format, **kwargs)

        elif isinstance(ee_object, ee.geometry.Geometry):
            geojson = ee_object.toGeoJSON()
            gdf = gpd.read_file(json.dumps(geojson), driver="GeoJSON")
            ee_layer = viz_layer(data=gdf, **kwargs)

        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
            ee_layer = BitmapTileLayer(data=map_id_dict["tile_fetcher"].url_format, **kwargs)

        self.add_layer(ee_layer)

    def zoom_to_bounds(self, feat: Union[BaseLayer, List[BaseLayer], gpd.GeoDataFrame]):
        """
        Zooms the map to the bounds of a dataframe or layer.

        Parameters
        ----------
        feat : BaseLayer, List[lonboard.BaseLayer], gpd.GeoDataFrame
            The feature to zoom to
        """
        if feat is None:
            view_state = compute_view(self.layers)
        elif isinstance(feat, gpd.GeoDataFrame):
            bounds = feat.to_crs(4326).total_bounds
            bbox = Bbox(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3])

            centerLon = (bounds[0] + bounds[2]) / 2
            centerLat = (bounds[1] + bounds[3]) / 2

            view_state = {
                "longitude": centerLon,
                "latitude": centerLat,
                "zoom": bbox_to_zoom_level(bbox),
                "pitch": 0,
                "bearing": 0,
            }
        else:
            view_state = compute_view(feat)

        self.set_view_state(**view_state)

    def add_geotiff(
        self,
        path: str,
        zoom: bool = False,
        cmap: Union[str, mpl.colors.Colormap] = None,
        opacity: float = 0.7,
    ):
        """
        Adds a local geotiff to the map
        Note that since deck.gl tiff support is limited, this extracts the CRS/Bounds from the tiff
        and converts the image data in-memory to PNG

        Parameters
        ----------
        path : str
            The path to the local tiff
        zoom : bool
            Whether to zoom the map to the bounds of the tiff
        cmap: str or matplotlib.colors.Colormap
            The colormap to apply to the raster
        opacity: float
            The opacity of the overlay
        """
        with rasterio.open(path) as src:
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            rio_kwargs = src.meta.copy()
            rio_kwargs.update({"crs": "EPSG:4326", "transform": transform, "width": width, "height": height})

            # new
            bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)

            if cmap is None:
                im = [rasterio.band(src, i + 1) for i in range(src.count)]
            else:
                cmap = mpl.cm.get_cmap(cmap)
                rio_kwargs["count"] = 4
                im = rasterio.band(src, 1)[0].read()[0]
                im_min, im_max = np.nanmin(im), np.nanmax(im)
                im = np.rollaxis(cmap((im - im_min) / (im_max - im_min), bytes=True), -1)
                # TODO Handle Colorbar

            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**rio_kwargs) as dst:
                    for i in range(rio_kwargs["count"]):
                        rasterio.warp.reproject(
                            source=im[i],
                            destination=rasterio.band(dst, i + 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs="EPSG:4326",
                            resampling=rasterio.warp.Resampling.nearest,
                        )
                    height = dst.height
                    width = dst.width

                    data = dst.read(
                        out_dtype=rasterio.uint8,
                        out_shape=(rio_kwargs["count"], int(height), int(width)),
                        resampling=rasterio.enums.Resampling.bilinear,
                    )

                    with rasterio.io.MemoryFile() as outfile:
                        with outfile.open(
                            driver="PNG",
                            height=data.shape[1],
                            width=data.shape[2],
                            count=rio_kwargs["count"],
                            dtype=data.dtype,
                        ) as mempng:
                            mempng.write(data)
                        url = "data:image/png;base64," + base64.b64encode(outfile.read()).decode("utf-8")

                        layer = BitmapLayer(image=url, bounds=bounds, opacity=opacity)
                        self.add_layer(layer, zoom=zoom)

    @staticmethod
    def get_named_tile_layer(layer: str) -> BitmapTileLayer:
        # From Leafmap
        # https://github.com/opengeos/leafmap/blob/master/leafmap/basemaps.py
        xyz_tiles = {
            "OpenStreetMap": {
                "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                "attribution": "OpenStreetMap",
                "name": "OpenStreetMap",
                "max_requests": -1,
            },
            "ROADMAP": {
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",  # noqa
                "attribution": "Esri",
                "name": "Esri.WorldStreetMap",
            },
            "SATELLITE": {
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "attribution": "Esri",
                "name": "Esri.WorldImagery",
            },
            "TERRAIN": {
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                "attribution": "Esri",
                "name": "Esri.WorldTopoMap",
            },
            "HYBRID": {
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "attribution": "Esri",
                "name": "Esri.WorldImagery",
            },
        }

        layer = xyz_tiles.get(layer)
        if not layer:
            raise ValueError("string layer name must be in  {}".format(", ".join(xyz_tiles.keys())))
        return BitmapTileLayer(
            data=layer.get("url"),
            tile_size=layer.get("tile_size", 128),
            max_zoom=layer.get("max_zoom", None),
            min_zoom=layer.get("min_zoom", None),
            max_requests=layer.get("max_requests", None),
        )
