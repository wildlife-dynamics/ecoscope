import base64
import os
import time
import typing
import urllib
import warnings


import ee
import folium
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import rasterio
import selenium.webdriver
from branca.colormap import StepColormap
from branca.element import MacroElement, Template

from ecoscope.contrib.foliumap import Map
from ecoscope.analysis.speed import SpeedDataFrame

warnings.filterwarnings("ignore", "GeoSeries.isna", UserWarning)


class EcoMapMixin:
    def add_speedmap(
        self,
        trajectory: gpd.GeoDataFrame,
        classification_method: str = "equal_interval",
        num_classes: int = 6,
        speed_colors: typing.List = None,
        bins: typing.List = None,
        legend: bool = True,
    ):

        speed_df = SpeedDataFrame.from_trajectory(
            trajectory=trajectory,
            classification_method=classification_method,
            num_classes=num_classes,
            speed_colors=speed_colors,
            bins=bins,
        )
        self.add_gdf(speed_df, color=speed_df["speed_colour"])

        if legend:
            self.add_legend(legend_dict=dict(zip(speed_df.label, speed_df.speed_colour)))

        return speed_df


class EcoMap(EcoMapMixin, Map):
    def __init__(self, *args, static=False, print_control=True, **kwargs):
        kwargs["attr"] = kwargs.get("attr", " ")
        kwargs["canvas"] = kwargs.get("canvas", True)
        kwargs["control_scale"] = kwargs.get("control_scale", False)
        kwargs["height"] = kwargs.get("height", 600)
        kwargs["width"] = kwargs.get("width", 800)

        if static:
            print_control = False
            kwargs["draw_control"] = kwargs.get("draw_control", False)
            kwargs["fullscreen_control"] = kwargs.get("fullscreen_control", False)
            kwargs["layers_control"] = kwargs.get("layers_control", False)
            kwargs["measure_control"] = kwargs.get("measure_control", False)
            kwargs["zoom_control"] = kwargs.get("zoom_control", False)
            kwargs["search_control"] = kwargs.get("search_control", False)

        self.px_height = kwargs["height"]
        self.px_width = kwargs["width"]

        super().__init__(*args, **kwargs)

        if print_control:
            self.add_print_control()

    def add_gdf(self, data, *args, simplify_tolerance=None, **kwargs):
        """
        Wrapper for `geopandas.explore._explore`.
        """

        data = data.copy()
        data = data.to_crs(4326)
        data = data.loc[(~data.geometry.isna()) & (~data.geometry.is_empty)]

        if simplify_tolerance is not None:
            data = data.simplify(simplify_tolerance)

        if isinstance(data, gpd.GeoDataFrame):
            for col in data:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    data[col] = data[col].astype("string")

        kwargs["m"] = self
        kwargs["tooltip"] = kwargs.get("tooltip", False)
        gpd.explore._explore(data, *args, **kwargs)

    def add_legend(self, *args, **kwargs):
        """
        Patched method for allowing legend hex colors to start with a "#".
        """
        legend_dict = kwargs.get("legend_dict")
        if legend_dict is not None:
            kwargs["legend_dict"] = {
                k: v[1:] if isinstance(v, str) and v.startswith("#") else v for k, v in legend_dict.items()
            }

        return super().add_legend(*args, **kwargs)

    def add_north_arrow(self, imagePath="", position="topright", angle=0, scale=1.0):
        """
        Parameters
        ----------
        imagePath : str
            Path to the image file for the north arrow. If not provided, default SVG image is uploaded.
        position : str
            Possible values are 'topleft', 'topright', 'bottomleft' or 'bottomright'.
        angle : int
            Angle for the north arrow to be rotated.
        scale : float
            Scale dimensions of north arrow.

        """

        self.add_child(
            NorthArrowElement(
                html=f"""\
            <svg width="{100*scale}px" height="{100*scale}px" version="1.1" viewBox="0 0 773 798" xmlns="http://www.w3.org/2000/svg">
<path transform="translate(0 798) scale(1 -1)" d="m674 403-161 48q-17 48-66 70l-46 166-46-167q-22-9-38-25t-29-45l-159-47 159-49q15-44 67-68l46-164 48 164q39 17 64 69zm-163 0q0-49-32-81-33-34-78-34-46 0-77 34-31 31-31 81 0 46 31 80t77 34q45 0 78-34 32-34 32-80zm-12 1q-5 7-7.5 17.5t-4 21.5-4.5 21-9 16q-7 6-17 9.5t-20.5 6-20 6-15.5 9.5v-107h98zm-98-108v108h-99l3-3 23-75q6-6 16.5-9.5t21-5.5 20-5.5 15.5-9.5zm-280 152h-26v-2q5 0 6-1 3-3 3-6 0-2-0.5-4t-1.5-7l-18-48-16 47q-3 9-3 12 0 7 7 7h2v2h-34v-2q2 0 3-1l3-3q2 0 2-2 2-1 4-5l5-15-12-42-17 50q-3 9-3 11 0 7 6 7h2v2h-33v-2q8 0 10-6 1-2 3-9l27-74h5l15 53 19-53h2l27 71q2 10 3 11 5 7 10 7v2zm325 350h-29v-3q7 0 10-4 1-1 1-11v-35l-42 53h-32v-3q7-2 12-6l2-3v-62q0-13-12-13v-2h29v2h-2q-4 0-7 2.5t-3 10.5v55l58-72h3v73q0 9 1 10.5t8 3.5l3 1v3zm207-395h-130q0 16-6 42zm-212-119-40-141v135q9 0 19 1t21 5zm-154 78-137 41h130q0-10 2-19.5t5-21.5zm114 168q-25 0-39-8l39 142v-134zm372-148h-3q-3-4-5-7.5t-4-5.5q-5-5-17-5h-19q-3 0-3 5v35h20q8 0 10-6 1-1 1-3 0-3 1-4h3v30h-3q-2-9-4-11t-8-2h-20v35h24q7 0 8-1 4-1 9-14h3l-1 20h-69v-2h3q7 0 8-4 2-2 2-9v-58q0-11-4-12-1-1-6-1h-3v-3h68zm-340-358q0 9-5.5 14.5t-20.5 14.5q-9 5-13 9l-5 5q-3 10-3 7 0 14 14 14 18 0 24-26h2v31h-2q-2-6-5-6-4 0-5 1-8 5-15 5-11 0-17.5-7t-6.5-17q0-13 9-19 6-4 16.5-10.5t12.5-8.5q8-7 8-13 0-14-18-14-13 0-18 5.5t-7 20.5h-2v-30h2q0 5 3 5l16-5h8q12 0 20 7t8 17z"/>
            </svg>""",  # noqa
                imagePath=imagePath,
                position=position,
                angle=angle,
                scale=scale,
            )
        )

    def add_scale_bar(self, style="single", position="bottomleft", imperial=False, force=False):
        """
        Parameters
        ----------
        style : str, optional
            Possible values are 'single', 'double', 'empty' or 'comb'. Default is 'single'.
        position : str, optional
            Possible values are 'topleft', 'topright', 'bottomleft' or 'bottomright'.
        imperial : bool, optional
            If True, the scale bar uses miles and feet. Default is False.
        force : bool, optional
            If True, adds index=0 to the add_child() call, forcing the scale bar 'behind' Folium's default controls
        """

        svg = (
            """<svg xmlns="http://www.w3.org/2000/svg" id="test" style="width:300;height:40px;"><rect id="border" style="stroke:#000;fill:#FFF;" height="40%" width="75%" x="5%" y="2%"/><rect id="first_block" style="fill:#000" height="20%" width="37.5%" x="5%" y="2%"/><rect id="second_block" style="fill:#000" height="20%" width="37.5%" x="42.5%" y="22%"/><text id="zero" text-anchor="middle" font-size="20" x="5%" y="95%">0</text><text id="half_scale" font-size="20" text-anchor="middle" x="42.5%" y="95%">15</text><text id="scale" font-size="20" text-anchor="middle" x="80%" y="95%">30</text><text id="unit" font-size="20" x="82%" y="42%">km</text></svg>"""  # noqa
            if style == "double"
            else """<svg xmlns="http://www.w3.org/2000/svg" id="test" style="width:300;height:40px;"><rect id="border" style="stroke:#000;fill:#FFF;" height="50%" width="75%" x="5%" y="48%" /><text id="zero" font-size="20" x="0%" y="95%" visibility="hidden">0</text><text id="half_scale" font-size="20" text-anchor="middle" x="42.5%" y="95%" visibility="hidden">15</text><text id="scale" font-size="20" text-anchor="middle" x="42.5%" y="92%">30</text><text id="unit" font-size="20" x="82%" y="95%">km</text></svg>"""  # noqa
            if style == "empty"
            else """<svg xmlns="http://www.w3.org/2000/svg" id="test" style="width:300;height:40px;"><svg y="50%" x="5%" width="75%" height="48%" viewBox="-0.4 0 101 101" preserveAspectRatio="none"><path d="M 0 50 L 0 99 h 50 V 50 V 99 H 100 V 50" vector-effect="non-scaling-stroke" fill="None" stroke-width="2" stroke="#000"/></svg><text id="zero" font-size="20" text-anchor="middle" x="5%" y="50%">0</text><text id="half_scale" font-size="20" text-anchor="middle" y="50%" x="42.5%">15</text><text id="scale" font-size="20" text-anchor="middle" y="50%" x="80%">1500</text><text id="unit" font-size="20" x="85%" y="98%">km</text></svg>"""  # noqa
            if style == "comb"
            else """<svg xmlns="http://www.w3.org/2000/svg" id="test" style="width:300;height:40px;"><rect id="border" style="stroke:#000;fill:#FFF;" height="40%" width="75%" x="5%" y="2%"/><rect id="first_block" style="fill:#000" height="40%" width="37.5%" x="5%" y="2%"/><text id="zero" text-anchor="middle" font-size="20" x="5%" y="95%">0</text><text id="half_scale" font-size="20" text-anchor="middle" x="42.5%" y="95%">15</text><text id="scale" font-size="20" text-anchor="middle" x="80%" y="95%">30</text><text id="unit" font-size="20" x="82%" y="42%">km</text></svg>"""  # noqa
        )

        self.add_child(ScaleElement(svg, position=position, imperial=imperial), index=0 if force else None)

    def add_title(
        self,
        title: str,
        font_size: str = "32px",
        font_style: str = "normal",
        font_family: typing.Union[str, list] = "Helvetica",
        font_color: typing.Union[str, tuple] = "rgba(0,0,0,1)",
        position: dict = None,
        background_color: typing.Union[str, tuple] = "#FFFFFF99",
        outline: typing.Union[str, dict] = "0px solid rgba(0, 0, 0, 0)",
        **kwargs,
    ):
        """
        Parameters
        ----------
        title : str
            Text of title.
        font_size : str
            CSS font size that includes units.
            font_style : str
        font_family:str"Helvetica",
            Font family selection; Could be one or more separated by spaces.
        font_color : str
            Text color (css color property); supports rgb, rgba and hex string formats.
        position : dict|None
            Dict object with top, left and bottom margin values for the title container position.
            ex. {
                    "top": "10px",
                    "left": "25px",
                    "right": "0px",
                    "bottom": "0px"
                }
            All keys are optional in the dictionary (could be passed some of them as necessary).
            Values could be passed as px or accepted css metric. Default None.
        background_color : str
            Box background color; supports rgb, rgba and hex string formats. Default '#FFFFFF99'.
        outline : str
            Element outline values (width style color_with_transparency).
            Could be passed as a string with spaced separated values or a dict structure:
            ex. {
                    "width": "1px",
                    "style": "solid",
                    "color": "#FFFFFF99" # or rgb/rgba tuple (0, 0, 0, 0)
                }
        kwargs
            Additional style kwargs. Underscores in keys will be replaced with dashes
        """
        position_styles = ""
        if isinstance(position, dict):
            VALID_POSITION_OPTIONS = ["top", "bottom", "left", "right"]
            position_styles = " ".join([f"{k}:{v};" for k, v in position.items() if k in VALID_POSITION_OPTIONS])
        else:
            position_styles = "left: 50%;"
        if isinstance(font_family, str):
            font = font_family
        elif isinstance(font_family, list):
            font = " ".join(font_family)
        if isinstance(font_color, tuple):
            if font_color.__len__() == 3:
                fc = "rgb({},{},{})".format(font_color[0], font_color[1], font_color[2])
            elif font_color.__len__() == 4:
                fc = "rgba({},{},{},{})".format(font_color[0], font_color[1], font_color[2], font_color[3])
        elif isinstance(font_color, str):
            fc = font_color if font_color.startswith("#") else f"#{font_color}"
        if isinstance(background_color, tuple):
            if background_color.__len__() == 3:
                bkg = "rgb({},{},{})".format(background_color[0], background_color[1], background_color[2])
            elif background_color.__len__() == 4:
                bkg = "rgba({},{},{},{})".format(
                    background_color[0], background_color[1], background_color[2], background_color[3]
                )
        elif isinstance(background_color, str):
            bkg = background_color if background_color.startswith("#") else f"#{background_color}"
        outline = (
            outline
            if isinstance(outline, str)
            else "{} {} {}".format(outline["width"], outline["style"], outline["color"])
        )
        title_html = f"""\
        <div style="position: absolute; {position_styles}">
            <p style="position: relative;
                    left: -50%;
                    border: 1px solid #000;
                    border-radius: 5px;
                    background-color: {bkg};
                    padding: 3px;
                    outline: {outline};
                    font-size: { font_size };
                    font-family: {font};
                    font-style:{font_style};
                    color: {fc};
                    { "".join([f"{k.replace('_', '-')}: {v};" for k, v in kwargs.items() ]) }
                    ">
            { title }
            </p>
        </div>
        """
        self.add_child(FloatElement(title_html, top=0, left=0, right=0))

    def _repr_html_(self, **kwargs):
        if kwargs.get("fill_parent", False):
            original_width = self._parent.width
            original_height = self._parent.height

            self._parent.width = "100%"
            self._parent.height = "100%"

        html = (
            super()
            ._repr_html_(**kwargs)
            .replace(urllib.parse.quote("crs: L.CRS."), urllib.parse.quote("attributionControl: false, crs: L.CRS."))
        )

        # this covers the (probably) rare case where someone in a Jupyter setting:
        # creates a map, calls to_html(), continues changing the map, and then displays via _repr_html_()
        if kwargs.get("fill_parent", False):
            self._parent.width = original_width
            self._parent.height = original_height

        return html

    def to_html(self, outfile, fill_parent=True, **kwargs):
        """
        Parameters
        ----------
        outfile : str, Pathlike
            Output destination

        """
        with open(outfile, "w") as file:
            file.write(self._repr_html_(fill_parent=fill_parent, **kwargs))

    def to_png(self, outfile, sleep_time=5, **kwargs):
        """
        Parameters
        ----------
        outfile : str, Pathlike
            Output destination
        sleep_time : int, optional
            Additional seconds to wait before taking screenshot. Should be increased if map tiles in the output haven't
            fully loaded but can also be decreased in most cases.

        """
        tempfile = "tmp_to_png.html"
        super().to_html(tempfile, **kwargs)
        chrome_options = selenium.webdriver.chrome.options.Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = selenium.webdriver.Chrome(options=chrome_options)
        if self.px_width and self.px_height:
            driver.set_window_size(width=self.px_width, height=self.px_height)
        driver.get(f"file://{os.path.abspath(tempfile)}")
        time.sleep(sleep_time)
        driver.save_screenshot(outfile)
        os.remove(tempfile)

    def add_ee_layer(self, ee_object, visualization_params, name) -> None:
        """
        Method for displaying Earth Engine image tiles.

        Parameters
        ----------
        ee_object
        visualization_params
        name

        Returns
        -------
        None

        """

        try:
            if isinstance(ee_object, ee.image.Image):
                map_id_dict = ee.Image(ee_object).getMapId(visualization_params)
                folium.raster_layers.TileLayer(
                    tiles=map_id_dict["tile_fetcher"].url_format,
                    attr="Google Earth Engine",
                    name=name,
                    overlay=True,
                    control=True,
                ).add_to(self)

            elif isinstance(ee_object, ee.imagecollection.ImageCollection):
                ee_object_new = ee_object.mosaic()
                map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
                folium.raster_layers.TileLayer(
                    tiles=map_id_dict["tile_fetcher"].url_format,
                    attr="Google Earth Engine",
                    name=name,
                    overlay=True,
                    control=True,
                ).add_to(self)

            elif isinstance(ee_object, ee.geometry.Geometry):
                folium.GeoJson(data=ee_object.getInfo(), name=name, overlay=True, control=True).add_to(self)

            elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
                ee_object_new = ee.Image().paint(ee_object, 0, 2)
                map_id_dict = ee.Image(ee_object_new).getMapId(visualization_params)
                folium.raster_layers.TileLayer(
                    tiles=map_id_dict["tile_fetcher"].url_format,
                    attr="Google Earth Engine",
                    name=name,
                    overlay=True,
                    control=True,
                ).add_to(self)

        except Exception as exc:
            print(f"{exc}. Could not display {name}")

    def zoom_to_bounds(self, bounds):
        """Zooms to a bounding box in the form of [minx, miny, maxx, maxy].

        Parameters
        ----------
        bounds : [x1, y1, x2, y2]
            Map extent in WGS 84
        """

        assert -180 < bounds[0] <= bounds[2] < 180
        assert -90 < bounds[1] <= bounds[3] < 90

        self.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    def zoom_to_gdf(self, gs):
        """
        Parameters
        ----------
        gs : gpd.GeoSeries or gpd.GeoDataFrame
            Geometry to adjust map bounds to. CRS will be converted to WGS 84.

        """

        self.zoom_to_bounds(gs.geometry.to_crs(4326).total_bounds)

    def add_local_geotiff(self, path, zoom=False, cmap=None, colorbar=True):
        """
        Displays a local GeoTIFF.

        Parameters
        ----------
        path : str, Pathlike
            Path to local GeoTIFF
        zoom : bool
            Zoom to displayed GeoTIFF
        cmap : `matplotlib.colors.Colormap` or str or None
            Matplotlib colormap to apply to raster
        colorbar : bool
            Whether to add colorbar for provided `cmap`. Does nothing if `cmap` is not provided.
        """

        with rasterio.open(path) as src:
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({"crs": "EPSG:4326", "transform": transform, "width": width, "height": height})

            if cmap is None:
                im = [rasterio.band(src, i + 1) for i in range(src.count)]
            else:
                cmap = mpl.cm.get_cmap(cmap)
                kwargs["count"] = 4
                im = rasterio.band(src, 1)[0].read()[0]
                im_min, im_max = np.nanmin(im), np.nanmax(im)
                im = np.rollaxis(cmap((im - im_min) / (im_max - im_min), bytes=True), -1)
                if colorbar:
                    if isinstance(im_min, np.integer) and im_max - im_min < 256:
                        self.add_child(
                            StepColormap(
                                [mpl.colors.rgb2hex(color) for color in cmap(np.linspace(0, 1, 1 + im_max - im_min))],
                                index=np.arange(1 + im_max - im_min),
                                vmin=im_min,
                                vmax=im_max + 1,
                            )
                        )
                    else:
                        self.add_child(
                            StepColormap(
                                [mpl.colors.rgb2hex(color) for color in cmap(np.linspace(0, 1, 256))],
                                vmin=im_min,
                                vmax=im_max,
                            )
                        )

            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**kwargs) as dst:
                    for i in range(kwargs["count"]):
                        rasterio.warp.reproject(
                            source=im[i],
                            destination=rasterio.band(dst, i + 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs="EPSG:4326",
                            resampling=rasterio.warp.Resampling.nearest,
                        )

                url = "data:image/tiff;base64," + base64.b64encode(memfile.read()).decode("utf-8")

        self.add_child(GeoTIFFElement(url, zoom))

    def add_print_control(self):
        self.add_child(PrintControl())

    def add_pil_image(self, image, bounds, name=None, zoom=True, opacity=1):
        """
        Overlays a PIL.Image onto the Ecomap

        Parameters
        ----------
        image : PIL.Image
            The image to be overlaid
        bounds: tuple
            Tuple containing the EPSG:4326 (minx, miny, maxx, maxy) values bounding the given image
        name : string, optional
            The name of the image layer to be displayed in Folium layer control
        zoom : bool, optional
            Zoom to the generated image
        opacity : float, optional
            Sets opacity of overlaid image
        """

        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=opacity,
            mercator_project=True,
            name=name,
        ).add_to(self)

        if zoom:
            self.zoom_to_bounds(bounds)


class ControlElement(MacroElement):
    """
    Class to wrap arbitrary HTML as Leaflet Control.

    Parameters
    ----------
    html : str
        HTML to render an element from.
    position : str
        Possible values are 'topleft', 'topright', 'bottomleft' or 'bottomright'.

    """

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.Control.extend({
            onAdd: function(map) {
                var template = document.createElement('template');
                template.innerHTML = `{{ this.html }}`.trim();
                return template.content.firstChild;
            }
        });
        (new {{ this.get_name() }}({{ this.options|tojson }})).addTo({{this._parent.get_name()}});

        {% endmacro %}
    """
    )

    def __init__(self, html, position="bottomright"):
        super().__init__()
        self.html = html
        self.options = folium.utilities.parse_options(
            position=position,
        )


class NorthArrowElement(ControlElement):
    """
    Class to wrap arbitrary HTML or PNG as Leaflet Control.

    Parameters
    ----------
    html : str
        HTML to render an element from.
    imagePath: str
        Path to the image file for the north arrow. If not provided, default SVG image is uploaded.
    position : str
        Possible values are 'topleft', 'topright', 'bottomleft' or 'bottomright'.
    angle : int
        Angle for the north arrow to be rotated.
    scale : float
        Scale dimensions of north arrow.

    """

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.Control.extend({
            onAdd: function(map) {
                var control;
                if (this.options.imagePath){
                    var img = document.createElement('img');
                    img.src=this.options.imagePath;
                    img.style.width = (this.options.scale * 100) + 'px';
                    img.style.height = (this.options.scale * 100) + 'px';
                    control=img;
                } else {
                    var template = document.createElement('template');
                    template.innerHTML = `{{ this.html }}`.trim();
                    control = template.content.firstChild
                }

                control.style.transform = 'rotate({{ this.angle }}deg)';
                return control;
            }
        });
        (new {{ this.get_name() }}({{ this.options|tojson }})).addTo({{this._parent.get_name()}});

        {% endmacro %}
    """
    )

    def __init__(self, html, imagePath="", position="bottomright", angle="", scale=""):
        super().__init__(html=html, position=position)
        self.angle = angle
        self.options = folium.utilities.parse_options(imagePath=imagePath, scale=scale, position=position)


class ScaleElement(MacroElement):
    """
    Class to wrap arbitrary HTML as Leaflet Control.

    Parameters
    ----------
    html : str
        HTML to render an element from.
    position : str
        Possible values are 'topleft', 'topright', 'bottomleft' or 'bottomright'.

    """

    _template = Template(
        """
        {% macro script(this, kwargs) %}

        var {{ this.get_name() }} = L.Control.Scale.extend({
            onAdd: function(map) {

                var container = document.createElement('div');
                container.classList.add("leaflet-control-scale");
                container.innerHTML = `{{ this.html }}`.trim();

                this._scale = container.firstChild;

                map.on('move', this._update, this);
                map.whenReady(this._update, this);

                return container;
            },

            _updateImperial(maxMeters) {
                const maxFeet = maxMeters * 3.2808399;
                let maxMiles, miles, feet;

                if (maxFeet > 5280) {
                    maxMiles = maxFeet / 5280;
                    miles = this._getRoundNum(maxMiles);
                    this._updateScale(this._scale, miles, "mi", miles / maxMiles);

                } else {
                    feet = this._getRoundNum(maxFeet);
                    this._updateScale(this._scale, feet, "ft", feet / maxFeet);
                }
            },

            _updateMetric(maxMeters) {
                const meters = this._getRoundNum(maxMeters),
                label = meters < 1000 ? `${meters} m` : `${meters / 1000} km`;

                value = meters < 1000 ? meters : meters / 1000;
                unit = meters < 1000 ? `m` : `km`;

                this._updateScale(this._scale, value, unit, meters / maxMeters);
            },

            _updateScale(scale, value, unit, ratio) {
                scale.style.width = `${Math.round(this.options.maxWidth * ratio * (4/3))}px`;

                scale.getElementById("scale").textContent = value;
                scale.getElementById("half_scale").textContent = value / 2;
                scale.getElementById("unit").textContent = unit;

            },

            onRemove(map) {
                map.off('move', this._update, this);
            }

        });
        (new {{ this.get_name() }}({{ this.options|tojson }})).addTo({{this._parent.get_name()}});

        {% endmacro %}
    """
    )

    def __init__(self, html, position="bottomright", imperial=False):
        super().__init__()
        self.html = html
        self.options = folium.utilities.parse_options(
            position=position, maxWidth=300, imperial=imperial, metric=not imperial
        )


class FloatElement(MacroElement):
    """
    Class to wrap arbitrary HTML as a floating Element.

    Parameters
    ----------
    html : str
        HTML to render an element from.
    left, right, top, bottom : str
        Distance between edge of map and nearest edge of element. Two should be provided. Style can also be specified
        in html.

    """

    _template = Template(
        """
        {% macro header(this,kwargs) %}
        <style>
            #{{ this.get_name() }} {
                position: absolute;
                z-index: 9999;
                left: {{ this.left }};
                right: {{ this.right }};
                top: {{ this.top }};
                bottom: {{ this.bottom }};
            }
        </style>
        {% endmacro %}

        {% macro html(this,kwargs) %}
        <div id="{{ this.get_name() }}">
             {{ this.html }}
        </div>
        {% endmacro %}
    """
    )

    def __init__(self, html, left="", right="", top="", bottom=""):
        super().__init__()
        self.html = html
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


class GeoTIFFElement(MacroElement):
    """
    Class to display a GeoTIFF.

    Parameters
    ----------
    url : str
        URL of GeoTIFF
    zoom : bool
        Zoom to displayed GeoTIFF
    """

    _template = Template(
        """
    {% macro html(this, kwargs) %}
        <script src="https://unpkg.com/georaster-layer-for-leaflet@3.8.0/dist/georaster-layer-for-leaflet.min.js"></script>
        <script src="https://unpkg.com/georaster@1.5.6/dist/georaster.browser.bundle.min.js"></script>
    {% endmacro %}

    {% macro script(this, kwargs) %}
        fetch(`{{ this.url }}`)
        .then(response => response.arrayBuffer())
        .then(arrayBuffer => {
          parseGeoraster(arrayBuffer).then(georaster => {
            var layer = new GeoRasterLayer({
                georaster: georaster,
                opacity: 0.7
            });
            layer.addTo({{this._parent.get_name()}});

            if ("True" == `{{ this.zoom }}`) {
                ({{ this._parent.get_name() }}).fitBounds(layer.getBounds());
            }

        });
        });
    {% endmacro %}
    """  # noqa
    )

    def __init__(self, url, zoom=False):
        super().__init__()
        self.url = url
        self.zoom = zoom


class PrintControl(MacroElement):
    _template = Template(
        """
                {% macro header(this, kwargs) %}
                <link rel="stylesheet" href="https://unpkg.com/leaflet.bigimage@1.0.1/src/Leaflet.BigImage.css">
                <script src='https://unpkg.com/leaflet.bigimage@1.0.1/src/Leaflet.BigImage.js'></script>
                {% endmacro %}
                {% macro script(this, kwargs) %}
                    L.control.BigImage().addTo(({{ this._parent.get_name() }}));
                {% endmacro %}
        """  # noqa
    )
