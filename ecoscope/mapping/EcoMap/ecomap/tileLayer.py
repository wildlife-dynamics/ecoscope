import traitlets
from ecomap.layer import Layer


class TileLayer(Layer):

    _layer_type = traitlets.Unicode("TileLayer").tag(sync=True)
    data = traitlets.Union([traitlets.Unicode(), traitlets.List(traitlets.Unicode(), minlen=1)]).tag(sync=True)
    min_zoom = traitlets.Int(default_value=0).tag(sync=True)
    max_zoom = traitlets.Int(default_value=None, allow_none=True).tag(sync=True)
    tile_size = traitlets.Int(default_value=512).tag(sync=True)

    # todo: extents and max_requests
    # max_requests will be useful with defined 'basemaps'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
