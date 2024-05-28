import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "plot": [
            "EcoPlotData",
            "add_seasons",
            "ecoplot",
            "mcp",
            "nsd",
            "plot_seasonal_dist",
            "speed",
        ],
    },
)

__all__ = [
    "EcoPlotData",
    "add_seasons",
    "ecoplot",
    "mcp",
    "nsd",
    "speed",
    "plot_seasonal_dist",
]
