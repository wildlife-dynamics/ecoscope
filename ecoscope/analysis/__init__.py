import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["UD", "seasons", "speed"],
    submod_attrs={
        "ecograph": ["Ecograph", "get_feature_gdf"],
        "percentile": ["get_percentile_area"],
    },
)

__all__ = [
    "Ecograph",
    "SpeedDataFrame",
    "UD",
    "get_feature_gdf",
    "get_percentile_area",
    "seasons",
]
