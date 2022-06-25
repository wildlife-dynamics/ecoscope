from ecoscope import analysis, base, contrib, io, mapping, plotting

ASCII = """\
 _____
|   __|___ ___ ___ ___ ___ ___ ___
|   __|  _| . |_ -|  _| . | . | -_|
|_____|___|___|___|___|___|  _|___|
                          |_|
"""

__initialized = False


def init(silent=False, selenium=False, force=False):
    """
    Initializes the environment with ecoscope-specific customizations.

    Parameters
    ----------
    silent : bool, optional
        Removes console output
    selenium : bool, optional
        Installs selenium webdriver in a colab environment
    force : bool, optional
        Ignores `__initialized`

    """

    global __initialized
    if __initialized and not force:
        if not silent:
            print("Ecoscope already initialized.")
        return

    import pandas as pd

    pd.options.plotting.backend = "plotly"

    from tqdm.auto import tqdm

    tqdm.pandas()

    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings(action="ignore", category=ShapelyDeprecationWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

    import geopandas as gpd

    def explore(data, *args, **kwargs):
        """
        Monkey-patched `geopandas.explore._explore` to use EcoMap instead.
        """
        initialized = "m" in kwargs
        if not initialized:
            kwargs["m"] = mapping.EcoMap()

        if isinstance(kwargs["m"], mapping.EcoMap):
            m = kwargs.pop("m")
            m.add_gdf(data, *args, **kwargs)
            if not initialized:
                m.zoom_to_bounds(data.geometry.to_crs(4326).total_bounds)
            return m
        else:
            return gpd.explore._explore(data, *args, **kwargs)

    gpd.GeoDataFrame.explore = explore
    gpd.GeoSeries.explore = explore

    import plotly.io as pio

    pio.templates.default = "seaborn"

    import sys

    if "google.colab" in sys.modules and selenium:
        import subprocess

        subprocess.run(["apt", "update"])
        subprocess.run(["apt", "install", "chromium-chromedriver"])

    __initialized = True
    if not silent:
        print(ASCII)


__all__ = [
    "analysis",
    "base",
    "contrib",
    "init",
    "io",
    "mapping",
    "plotting",
]
