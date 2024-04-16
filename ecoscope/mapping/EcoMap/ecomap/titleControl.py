import traitlets
from ecomap.mapControl import MapControl


class TitleControl(MapControl):
    _control_type = traitlets.Unicode("TitleControl").tag(sync=True)

    title = traitlets.Unicode(default_value=None).tag(sync=True)

    font_size = traitlets.Unicode(default_value="32px").tag(sync=True)
    font_style = traitlets.Unicode(default_value="normal").tag(sync=True)
    font_family = traitlets.Unicode(default_value="Helvetica").tag(sync=True)
    font_color = traitlets.Unicode(default_value="rgba(0,0,0,1)").tag(sync=True)
    background_color = traitlets.Unicode(default_value="rgba(255, 255, 255, 0.6)").tag(sync=True)
    outline = traitlets.Unicode(default_value="0px solid rgba(0, 0, 0, 0)").tag(sync=True)
    border_radius = traitlets.Unicode(default_value="5px").tag(sync=True)
    border = traitlets.Unicode(default_value="1px solid rgba(0, 0, 0, 1)").tag(sync=True)
    padding = traitlets.Unicode(default_value="3px").tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
