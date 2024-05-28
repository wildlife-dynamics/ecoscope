import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "map": [
            "ControlElement",
            "EcoMap",
            "FloatElement",
            "NorthArrowElement",
            "ScaleElement",
            "GeoTIFFElement",
            "PrintControl",
        ],
    },
)

__all__ = [
    "ControlElement",
    "EcoMap",
    "FloatElement",
    "NorthArrowElement",
    "ScaleElement",
    "GeoTIFFElement",
    "PrintControl",
]
