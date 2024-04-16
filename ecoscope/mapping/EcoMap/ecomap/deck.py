import os
import traitlets
import ee
from anywidget import AnyWidget
from pathlib import Path
from ecomap.layer import Layer
from ecomap.tileLayer import TileLayer
from ecomap.mapControl import MapControl
from ecomap.geoJsonLayer import GeoJsonLayer
from ipywidgets import widget_serialization


# bundler yields EcoMap/static/{index.js,styles.css}
bundler_output_dir = Path(__file__).parent / "static"


class Map(AnyWidget):
    _esm = os.path.join(bundler_output_dir, "index.js")
    # _css = os.path.join(bundler_output_dir, "style.css")

    width = traitlets.Int(default_value=800).tag(sync=True)
    height = traitlets.Int(default_value=600).tag(sync=True)

    # unclear if we need this here, or can be shoveled into ts
    controller = traitlets.Bool(default_value=True).tag(sync=True)

    initial_view_state = traitlets.Dict({}).tag(sync=True)  # make me a class

    # views = traitlets.List(trait=traitlets.Instance(View))
    # make layer base
    layers = traitlets.List(trait=traitlets.Instance(Layer)).tag(sync=True, **widget_serialization)
    controls = traitlets.List(trait=traitlets.Instance(MapControl)).tag(sync=True, **widget_serialization)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add_layer(self, layer: Layer):
        # replace this with spectate most likely
        update = self.layers.copy()
        update.append(layer)
        self.layers = update

    def add_tile_layer(self, **kwargs):
        self.add_layer(TileLayer(**kwargs))

    def add_ee_layer(self, ee_object, visualization_params, **kwargs):
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(visualization_params)
            ee_layer = TileLayer(data=map_id_dict["tile_fetcher"].url_format, **kwargs)

        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
            ee_layer = TileLayer(data=map_id_dict["tile_fetcher"].url_format ** kwargs)

        elif isinstance(ee_object, ee.geometry.Geometry):
            ee_layer = GeoJsonLayer(data=ee_object.getInfo(), **kwargs)

        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
            ee_layer = TileLayer(data=map_id_dict["tile_fetcher"].url_format ** kwargs)

        self.add_layer(ee_layer)
