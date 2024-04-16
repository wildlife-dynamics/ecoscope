import traitlets
from ipywidgets import Widget


class MapControl(Widget):

    props = traitlets.Dict({}).tag(sync=True)  # make me a class
    placement = traitlets.Unicode(default_value=None, allow_none=True).tag(sync=True)  # needs validation
    viewId = traitlets.Unicode(default_value=None, allow_none=True).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
