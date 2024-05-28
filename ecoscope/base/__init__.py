import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_dataclasses": [
            "RelocsCoordinateFilter",
            "RelocsDateRangeFilter",
            "RelocsDistFilter",
            "RelocsSpeedFilter",
            "TrajSegFilter",
        ],
        "base": ["EcoDataFrame", "Relocations", "Trajectory"],
        "utils": ["cachedproperty", "create_meshgrid", "groupby_intervals", "to_EarthLocation", "is_night"],
    },
)

__all__ = [
    "EcoDataFrame",
    "Relocations",
    "RelocsCoordinateFilter",
    "RelocsDateRangeFilter",
    "RelocsDistFilter",
    "RelocsSpeedFilter",
    "TrajSegFilter",
    "Trajectory",
    "cachedproperty",
    "create_meshgrid",
    "groupby_intervals",
    "to_EarthLocation",
    "is_night",
]
