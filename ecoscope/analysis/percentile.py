import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import rasterio  # type: ignore[import-untyped]
import rasterio.features  # type: ignore[import-untyped]
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon

from ecoscope.io import raster


def _multipolygon(shapes, percentile):
    return MultiPolygon([shape(geom) for geom, value in shapes if np.isclose(value, percentile)])


def get_percentile_area(
    percentile_levels: list[int],
    raster_data: raster.RasterData,
    subject_id: str = "",
) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    percentile_levels: list[int]
        list of k-th percentile scores.
    raster_data: raster.RasterData
        array of raster values
    subject_id: str
        unique identifier for the subject

    Returns
    -------
    GeoDataFrame

    """
    shapes = []
    for percentile in percentile_levels:
        data_array = raster_data.data.copy()

        # calculate percentile value
        values = np.sort(data_array[~np.isnan(data_array)]).flatten()
        if len(values) == 0:
            continue

        csum = np.cumsum(values)
        percentile_val = values[np.argmin(np.abs(csum[-1] * (1 - percentile / 100) - csum))]

        # TODO: make a more explicit comparison for less than and greater than

        # Set any vals less than the cutoff to be nan
        data_array[data_array < percentile_val] = np.nan

        # Mask any vals that are less than the cutoff percentile
        data_array[data_array >= percentile_val] = percentile

        shapes.extend(rasterio.features.shapes(data_array, transform=raster_data.transform))

    return gpd.GeoDataFrame(
        [
            [subject_id, percentile, _multipolygon(shapes, percentile)]
            for percentile in sorted(percentile_levels, reverse=True)
        ],
        columns=["subject_id", "percentile", "geometry"],
        crs=raster_data.crs,
    )
