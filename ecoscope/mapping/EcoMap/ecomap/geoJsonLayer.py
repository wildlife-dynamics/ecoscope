import traitlets
from ecomap.layer import Layer


class GeoJsonLayer(Layer):

    _layer_type = traitlets.Unicode("GeoJsonLayer").tag(sync=True)
    data = traitlets.Unicode().tag(sync=True)
    point_type = traitlets.Unicode(default_value="circle").tag(sync=True)

    # todo: add style options

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
