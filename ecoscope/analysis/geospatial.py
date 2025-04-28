from typing import Literal, Tuple

import geopandas as gpd  # type: ignore[import-untyped]
from PIL.Image import Image  # type: ignore[import-untyped]

from ecoscope.base.utils import BoundingBox

try:
    import datashader as ds  # type: ignore[import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["analysis"]'
    )


def datashade_gdf(
    gdf: gpd.GeoDataFrame,
    geom_type: Literal["polygon", "line", "point"],
    width: int = 600,
    height: int = 600,
    cmap: str | list[str] | None = None,
    ds_agg: ds.reductions.Reduction | None = None,
    **kwargs,
) -> Tuple[Image, BoundingBox]:
    """
    Creates a raster of the given gdf using Datashader

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame used to create visualization (geometry must be projected to a CRS)
    geom_type : str
        The Datashader canvas() function to use valid values are 'polygon', 'line', 'point'
    width : int
        The canvas width in pixels, determines the resolution of the generated image
    height : int
        The canvas height in pixels, determines the resolution of the generated image
    cmap : list of colors or matplotlib.colors.Colormap
        The colormap to use for the generated image
    ds_agg : datashader.reductions function
        The Datashader reduction to use
    kwargs
        Additional kwargs passed to datashader.transfer_functions.shade

    Returns
    ----------
    A tuple containing a PIL.Image raster and its EPSG:4326 bounds
    """

    gdf = gdf.to_crs(epsg=4326)
    bounds = gdf.geometry.total_bounds
    canvas = ds.Canvas(width, height)

    if cmap is None:
        cmap = ["lightblue", "darkblue"]

    if geom_type == "polygon":
        func = canvas.polygons
    elif geom_type == "line":
        func = canvas.line
    elif geom_type == "point":
        func = canvas.points
    else:
        raise ValueError("geom_type must be 'polygon', 'line' or 'point'")

    agg = func(gdf, geometry="geometry", agg=ds_agg)
    img = ds.tf.shade(agg, cmap, **kwargs)

    return img.to_pil(), bounds
