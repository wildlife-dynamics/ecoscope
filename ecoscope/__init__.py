from ecoscope import base, io

ASCII = """\
 _____
|   __|___ ___ ___ ___ ___ ___ ___
|   __|  _| . |_ -|  _| . | . | -_|
|_____|___|___|___|___|___|  _|___|
                          |_|
"""

__initialized = False


def init(silent=False, force=False):
    """
    Initializes the environment with ecoscope-specific customizations.

    Parameters
    ----------
    silent : bool, optional
        Removes console output
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

    import plotly.io as pio  # type: ignore[import-untyped]

    pio.templates.default = "seaborn"

    __initialized = True
    if not silent:
        print(ASCII)


__all__ = ["analysis", "base", "contrib", "init", "io", "mapping", "plotting"]
