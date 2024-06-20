import ee
import base64
import rasterio
import geopandas as gpd
import matplotlib as mpl
import numpy as np
from typing import List, Union
from lonboard import Map
from lonboard._geoarrow.ops.bbox import Bbox
from lonboard._viewport import compute_view, bbox_to_zoom_level
from lonboard._layer import BaseLayer, BaseArrowLayer, BitmapLayer, BitmapTileLayer
from lonboard._deck_widget import (
    BaseDeckWidget,
    NorthArrowWidget,
    ScaleWidget,
    LegendWidget,
    TitleWidget,
    SaveImageWidget,
)


class EcoMap2(Map):
    def __init__(self, static=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_layer(self, layer: BaseLayer):
        update = self.layers.copy()
        update.append(layer)
        self.layers = update

    def add_widget(self, widget: BaseDeckWidget):
        update = self.deck_widgets.copy()
        update.append(widget)
        self.deck_widgets = update

    def add_gdf(self, gdf: gpd.GeoDataFrame, **kwargs):
        self.add_layer(BaseArrowLayer.from_geopandas(gdf=gdf, **kwargs))

    def add_legend(self, **kwargs):
        self.add_widget(LegendWidget(**kwargs))

    def add_north_arrow(self, **kwargs):
        self.add_widget(NorthArrowWidget(**kwargs))

    def add_scale_bar(self, **kwargs):
        self.add_widget(ScaleWidget(**kwargs))

    def add_title(self, **kwargs):
        self.add_widget(TitleWidget(**kwargs))

    def add_save_image(self, **kwargs):
        self.add_widget(SaveImageWidget(**kwargs))

    def add_ee_layer(self, ee_object, visualization_params, **kwargs):
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(visualization_params)
            ee_layer = BitmapTileLayer(data=map_id_dict["tile_fetcher"].url_format, **kwargs)

        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
            ee_layer = BitmapTileLayer(data=map_id_dict["tile_fetcher"].url_format, **kwargs)

        elif isinstance(ee_object, ee.geometry.Geometry):
            gdf = gpd.GeoDataFrame([ee_object.toGeoJSON()])
            ee_layer = BaseArrowLayer.from_geopandas(gdf=gdf, **kwargs)

        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
            ee_layer = BitmapTileLayer(data=map_id_dict["tile_fetcher"].url_format, **kwargs)

        self.add_layer(ee_layer)

    def zoom_to_bounds(self, feat: Union[BaseLayer, List[BaseLayer], gpd.GeoDataFrame]):
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
        colorbar: bool = True,
        opacity: float = 0.7,
    ):
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
                # if colorbar:
                #     if isinstance(im_min, np.integer) and im_max - im_min < 256:
                #         self.add_child(
                #             StepColormap(
                #                 [mpl.colors.rgb2hex(color) for color in cmap(np.linspace(0, 1, 1 + im_max - im_min))],
                #                 index=np.arange(1 + im_max - im_min),
                #                 vmin=im_min,
                #                 vmax=im_max + 1,
                #             )
                #         )
                #     else:
                #         self.add_child(
                #             StepColormap(
                #                 [mpl.colors.rgb2hex(color) for color in cmap(np.linspace(0, 1, 256))],
                #                 vmin=im_min,
                #                 vmax=im_max,
                #             )
                #         )

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
                        self.add_layer(layer)
