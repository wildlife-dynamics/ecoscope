import traitlets
from ipywidgets import Widget


class Layer(Widget):

    layer_name = traitlets.Unicode(default_value=None, allow_none=True).tag(sync=True)
    visible = traitlets.Bool(default_value=True).tag(sync=True)
    opacity = traitlets.Float(default_value=1, min=0, max=1).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
