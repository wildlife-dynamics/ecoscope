from ecoscope import base, io

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

    # Enable copy-on-write for pandas. It will be the default in pandas 3.0.
    pd.options.mode.copy_on_write = True

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
        from ecoscope import mapping

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
        from IPython import get_ipython

        shell_text = """\
cat > /etc/apt/sources.list.d/debian.list <<'EOF'
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-bookworm.gpg] http://deb.debian.org/debian bookworm main
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-bookworm-updates.gpg]\
        http://deb.debian.org/debian bookworm-updates main
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-security-bookworm.gpg]\
        http://deb.debian.org/debian-security bookworm/updates main
EOF

apt-key adv --keyserver keyserver.ubuntu.com --recv-keys DCC9EFBF77E11517
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 112695A0E562B32A

apt-key export 77E11517 | gpg --dearmour -o /usr/share/keyrings/debian-bookworm.gpg
apt-key export 22F3D138 | gpg --dearmour -o /usr/share/keyrings/debian-bookworm-updates.gpg
apt-key export E562B32A | gpg --dearmour -o /usr/share/keyrings/debian-security-bookworm.gpg

cat > /etc/apt/preferences.d/chromium.pref << 'EOF'
Package: *
Pin: release a=eoan
Pin-Priority: 500


Package: *
Pin: origin "deb.debian.org"
Pin-Priority: 300


Package: chromium*
Pin: origin "deb.debian.org"
Pin-Priority: 700
EOF

apt-get update
apt-get install chromium chromium-driver

pip install selenium
"""

        if silent:
            from IPython.utils import io

            with io.capture_output():
                get_ipython().run_cell_magic("shell", "", shell_text)
        else:
            get_ipython().run_cell_magic("shell", "", shell_text)

    __initialized = True
    if not silent:
        print(ASCII)


__all__ = ["analysis", "base", "contrib", "init", "io", "mapping", "plotting"]
