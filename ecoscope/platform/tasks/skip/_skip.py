from typing import Any

from wt_registry import register
from wt_task.skip import SKIP_SENTINEL, SkipSentinel

from ecoscope.platform.annotations import AnyGeoDataFrame


def _is_df_duck(obj: Any) -> bool:
    """Check if an object quacks like a DataFrame-like object (via duck typing)."""
    return (
        hasattr(obj, "columns")
        and hasattr(obj, "index")
        and hasattr(obj, "iloc")
        and hasattr(obj, "loc")
        and hasattr(obj, "shape")
    )


def _is_gdf_duck(obj: Any) -> bool:
    """Check if an object quacks like a GeoDataFrame-like object (via duck typing)."""
    return _is_df_duck(obj) and hasattr(obj, "geometry")


@register()
def any_is_empty_df(*args: tuple[Any, ...]) -> bool:
    """Check if any item in the args is an empty DataFrame-like object."""
    return any(_is_df_duck(a) and getattr(a, "empty") for a in args)


@register()
def any_dependency_skipped(*args: tuple[Any, ...]) -> bool:
    """Check if any item in the iterable is the SKIP_SENTINEL,
    indicating that some dependency was skipped.
    """
    return any(item is SKIP_SENTINEL for item in args)


@register()
def never(*args: tuple[Any, ...]) -> bool:
    """Always return False."""
    return False


@register()
def all_keyed_iterables_are_skips(*args: tuple[Any, ...]) -> bool:
    """Check if all items in the keyed iterable are SKIP_SENTINEL."""
    return len([i for i in args for elem in i if elem[1] is not SKIP_SENTINEL]) == 0


@register()
def any_dependency_is_none(*args: tuple[Any, ...]) -> bool:
    """Check if any arg is None."""
    return any(item is None for item in args)


@register()
def any_dependency_is_empty_string(*args: tuple[Any, ...]) -> bool:
    """Check if any arg is an empty string."""
    return any(isinstance(item, str) and item == "" for item in args)


@register()
def all_geometry_are_none(*args: Any) -> bool:
    """Check if any item in the args is an GeoDataFrame-like object with a nulled out geometry column."""
    return any(_is_gdf_duck(a) and (a.geometry.isna() | a.geometry.is_empty).all() for a in args)


def skip_gdf_fallback_to_none(
    gdf: AnyGeoDataFrame | SkipSentinel,
) -> AnyGeoDataFrame | None:
    """Fallback function to convert SkipSentinel to None."""
    return None if isinstance(gdf, SkipSentinel) else gdf
