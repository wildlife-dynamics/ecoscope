from ecoscope.io import earthranger, eetools, raster, utils
from ecoscope.io.earthranger import EarthRangerIO
from ecoscope.io.utils import download_file

__all__ = [
    "earthranger",
    "EarthRangerIO",
    "download_file",
    "earthranger_utils",
    "eetools",
    "raster",
    "utils",
]

try:
    from ecoscope.io.async_earthranger import AsyncEarthRangerIO

    __all__ = [
        "earthranger",
        "EarthRangerIO",
        "AsyncEarthRangerIO",
        "download_file",
        "earthranger_utils",
        "eetools",
        "raster",
        "utils",
    ]
except ImportError:
    pass
