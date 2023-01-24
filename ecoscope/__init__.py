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
        import shlex
        import subprocess
      
        cat_1 = shlex.split("cat > /etc/apt/sources.list.d/debian.list <<'EOF'")
        subprocess.run(
            [
                cat_1,
                """
                 deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster.gpg] http://deb.debian.org/debian buster main
                 deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster-updates.gpg] http://deb.debian.org/debian buster-updates main
                 deb [arch=amd64 signed-by=/usr/share/keyrings/debian-security-buster.gpg] http://deb.debian.org/debian-security buster/updates main
                 EOF
                """,
            ])

        apt_keyadv_1 = shlex.split("apt-key adv --keyserver keyserver.ubuntu.com --recv-keys DCC9EFBF77E11517")
        subprocess.run(apt_keyadv_1)
        
        apt_keyadv_2 = shlex.split("apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138")
        subprocess.run(apt_keyadv_2)
        
        apt_keyadv_3 = shlex.split("apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 112695A0E562B32A")
        subprocess.run(apt_keyadv_3)

        apt_keyexport_1 = shlex.split("apt-key export 77E11517 | gpg --dearmour -o /usr/share/keyrings/debian-buster.gpg")
        subprocess.run(apt_keyexport_1)
        
        apt_keyexport_2 = shlex.split("apt-key export 22F3D138 | gpg --dearmour -o /usr/share/keyrings/debian-buster-updates.gpg")
        subprocess.run(apt_keyexport_2)
        
        apt_keyexport_3 = shlex.split("apt-key export E562B32A | gpg --dearmour -o /usr/share/keyrings/debian-security-buster.gpg")
        subprocess.run(apt_keyexport_3)

        cat_2 = shlex.split("cat > /etc/apt/preferences.d/chromium.pref << 'EOF'")
        subprocess.run(
            [
                cat_2,
                """
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
                """,
            ])
  
        apt_update = shlex.split("apt-get update")
        subprocess.run(apt_update)
        
        apt_install = shlex.split("apt-get install chromium chromium-driver")
        subprocess.run(apt_install)
        
        pip_install = shlex.split("pip install selenium")
        subprocess.run(pip_install)

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
