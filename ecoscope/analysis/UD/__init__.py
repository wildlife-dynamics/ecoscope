import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={"etd_range": ["calculate_etd_range"]},
)

__all__ = [
    "calculate_etd_range",
]
