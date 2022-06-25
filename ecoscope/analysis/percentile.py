import os
import typing
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon


@dataclass
class PercentileAreaProfile:
    input_raster: typing.Union[str, bytes, os.PathLike]
    percentile_levels: typing.List = field(default_factory=[50.0])
    subject_id: str = ""


class PercentileArea:
    @staticmethod
    def _multipolygon(shapes, percentile):
        return MultiPolygon([shape(geom) for geom, value in shapes if np.isclose(value, percentile)])

    @classmethod
    def calculate_percentile_area(cls, profile: PercentileAreaProfile):
        """
        Parameters
        ----------
        profile:  PercentileAreaProfile
            dataclass object with information for percentile-area calculation

        Returns
        -------
        GeodataFrame
        """

        assert type(profile) is PercentileAreaProfile
        shapes = []

        # open raster
        with rasterio.open(profile.input_raster) as src:
            crs = src.crs.to_wkt()

            for percentile in profile.percentile_levels:
                data_array = src.read(1).astype(np.float32)

                # Mask no-data values
                data_array[data_array == src.nodata] = np.nan

                # calculate percentile value
                # percentile_val = np.percentile(data_array[~np.isnan(data_array)], 100.0 - percentile)
                values = np.sort(data_array[~np.isnan(data_array)]).flatten()
                csum = np.cumsum(values)
                percentile_val = values[np.argmin(np.abs(csum[-1] * (1 - percentile / 100) - csum))]

                # TODO: make a more explicit comparison for less than and greater than

                # Set any vals less than the cutoff to be nan
                data_array[data_array < percentile_val] = np.nan

                # Mask any vals that are less than the cutoff percentile
                data_array[data_array >= percentile_val] = percentile

                shapes.extend(rasterio.features.shapes(data_array, transform=src.transform))

        return gpd.GeoDataFrame(
            [
                [profile.subject_id, percentile, cls._multipolygon(shapes, percentile)]
                for percentile in sorted(profile.percentile_levels, reverse=True)
            ],
            columns=["subject_id", "percentile", "geometry"],
            crs=crs,
        )


def get_percentile_area(
    percentile_levels: typing.List,
    raster_path: typing.Union[str, bytes, os.PathLike],
    subject_id: str = "",
):
    """
    Parameters
    ----------
    percentile_levels: Typing.List[Int]
        list of k-th percentile scores.
    raster_path: str or os.PathLike
        file path to where the raster is stored.
    subject_id: str
        unique identifier for the subject

    Returns
    -------
    GeoDataFrame

    """
    percentile_profile = PercentileAreaProfile(
        percentile_levels=percentile_levels,
        input_raster=raster_path,
        subject_id=subject_id,
    )
    return PercentileArea.calculate_percentile_area(profile=percentile_profile)
