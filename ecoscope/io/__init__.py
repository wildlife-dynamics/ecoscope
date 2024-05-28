import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["earthranger", "eetools", "raster", "utils"],
    submod_attrs={
        "earthranger": ["EarthRangerIO"],
        "utils": ["download_file"],
    },
)

__all__ = [
    "earthranger",
    "EarthRangerIO",
    "download_file",
    "eetools",
    "raster",
    "utils",
]
