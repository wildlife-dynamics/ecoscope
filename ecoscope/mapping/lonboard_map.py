import ee
import geopandas as gpd
from lonboard import Map
from lonboard._layer import BaseLayer, BaseArrowLayer, BitmapTileLayer
from lonboard._deck_widget import BaseDeckWidget, NorthArrowWidget, ScaleWidget, LegendWidget, TitleWidget


class EcoMap(Map):
    def __init__(self, static=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_layer(self, layer: BaseLayer):
        self.layers = self.layers.copy().append(layer)

    def add_widget(self, widget: BaseDeckWidget):
        self.deck_widgets = self.deck_widgets.copy().append(widget)

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
