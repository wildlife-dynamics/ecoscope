import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

# import shapely
from pyproj import Geod

from ecoscope.base._dataclasses import (
    RelocsCoordinateFilter,
    RelocsDateRangeFilter,
    RelocsDistFilter,
    RelocsSpeedFilter,
    # TrajSegFilter,
)
from functools import cached_property


@pd.api.extensions.register_dataframe_accessor("relocations")
class NewRelocations:
    """
    Relocation is a model for a set of fixes from a given subject.
    Because fixes are temporal, they can be ordered asc or desc. The additional_data dict can contain info
    specific to the subject and relocations: name, type, region, sex etc. These values are applicable to all
    fixes in the relocations array. If they vary, then they should be put into each fix's additional_data dict.
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self.gdf = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj, gpd.GeoDataFrame)

    @property
    def start_fixes(self):
        # unpack xy-coordinates of start fixes
        return self.gdf["geometry"].x, self.gdf["geometry"].y

    @property
    def end_fixes(self):
        # unpack xy-coordinates of end fixes
        return self.gdf["_geometry"].x, self.gdf["_geometry"].y

    @property
    def inverse_transformation(self):
        # use pyproj geodesic inverse function to compute vectorized distance & heading calculations
        return Geod(ellps="WGS84").inv(*self.start_fixes, *self.end_fixes)

    @property
    def heading(self):
        # Forward azimuth(s)
        forward_azimuth, _, _ = self.inverse_transformation
        forward_azimuth[forward_azimuth < 0] += 360
        return forward_azimuth

    @property
    def dist_meters(self):
        _, _, distance = self.inverse_transformation
        return distance

    @property
    def timespan_seconds(self):
        return (self.gdf["_fixtime"] - self.gdf["fixtime"]).dt.total_seconds()

    @property
    def speed_kmhr(self):
        return (self.dist_meters / self.timespan_seconds) * 3.6

    def from_gdf(self, groupby_col=None, time_col="fixtime", uuid_col=None, **kwargs):
        """
        Parameters
        ----------
        gdf : GeoDataFrame
            Observations data
        groupby_col : str, optional
            Name of `gdf` column of identities to treat as separate individuals. Usually `subject_id`. Default is
            treating the gdf as being of a single individual.
        time_col : str, optional
            Name of `gdf` column containing relocation times. Default is 'fixtime'.
        uuid_col : str, optional
            Name of `gdf` column of row identities. Used as index. Default is existing index.
        """
        # if kwargs.get("copy") is not False:
        #     gdf = gdf.copy()

        if groupby_col is None:
            if "groupby_col" not in self.gdf:
                self.gdf["groupby_col"] = 0
        else:
            self.gdf["groupby_col"] = self.gdf.loc[:, groupby_col]

        if time_col != "fixtime":
            self.gdf["fixtime"] = self.gdf.loc[:, time_col]

        if not pd.api.types.is_datetime64_any_dtype(self.gdf["fixtime"]):
            warnings.warn(
                f"{time_col} is not of type datetime64. Attempting to automatically infer format and timezone. "
                "Results may be incorrect."
            )
            self.gdf["fixtime"] = pd.to_datetime(self.gdf["fixtime"])

        if self.gdf["fixtime"].dt.tz is None:
            warnings.warn(f"{time_col} is not timezone aware. Assuming datetime are in UTC.")
            self.gdf["fixtime"] = self.gdf["fixtime"].dt.tz_localize(tz="UTC")

        if self.gdf.crs is None:
            warnings.warn("CRS was not set. Assuming geometries are in WGS84.")
            self.gdf.set_crs(4326, inplace=True)

        if uuid_col is not None:
            self.gdf.set_index(uuid_col, drop=False, inplace=True)

        self.gdf["junk_status"] = False

        default_cols = ["groupby_col", "fixtime", "junk_status", "geometry"]
        extra_cols = self.gdf.columns.difference(default_cols)
        extra_cols = extra_cols[~extra_cols.str.startswith("extra__")]

        assert self.gdf.columns.intersection("extra__" + extra_cols).empty, "Column names overlap with existing `extra`"

        self.gdf.rename(columns=dict(zip(extra_cols, "extra__" + extra_cols)), inplace=True)
        return self

    @staticmethod
    def _apply_speedfilter(relocations, fix_filter):
        with warnings.catch_warnings():
            """
            Note : This warning can be removed once the version of Geopandas is updated
            on Colab to the one that fixes this bug
            """
            warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs")
            result = relocations.gdf.assign(
                _fixtime=relocations.gdf["fixtime"].shift(-1),
                _geometry=relocations.gdf["geometry"].shift(-1),
                _junk_status=relocations.gdf["junk_status"].shift(-1),
            )[:-1]

        result["speed_kmhr"] = relocations.speed_kmhr

        result.loc[
            (~result["junk_status"]) & (~result["_junk_status"]) & (result["speed_kmhr"] > fix_filter.max_speed_kmhr),
            "junk_status",
        ] = True

        result.drop(
            ["_fixtime", "_geometry", "_junk_status", "speed_kmhr"],
            axis=1,
            inplace=True,
        )
        return result

    @staticmethod
    def _apply_distfilter(relocations, fix_filter):
        with warnings.catch_warnings():
            """
            Note : This warning can be removed once the version of Geopandas is updated
            on Colab to the one that fixes this bug
            """
            warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs")
            result = relocations.gdf.assign(
                _junk_status=relocations.gdf["junk_status"].shift(-1),
                _geometry=relocations.gdf["geometry"].shift(-1),
            )[:-1]

        _, _, distance_m = Geod(ellps="WGS84").inv(
            result["geometry"].x, result["geometry"].y, result["_geometry"].x, result["_geometry"].y
        )
        result["distance_km"] = distance_m / 1000

        result.loc[
            (~result["junk_status"]) & (~result["_junk_status"]) & (result["distance_km"] < fix_filter.min_dist_km)
            | (result["distance_km"] > fix_filter.max_dist_km),
            "junk_status",
        ] = True

        result.drop(["_geometry", "_junk_status", "distance_km"], axis=1, inplace=True)
        return result

    def apply_reloc_filter(self, fix_filter=None, inplace=False):
        """Apply a given filter by marking the fix junk_status based on the conditions of a filter"""

        if not self.gdf["fixtime"].is_monotonic_increasing:
            self.gdf.sort_values("fixtime", inplace=True)
        assert self.gdf["fixtime"].is_monotonic_increasing

        if inplace:
            frame = self.gdf
        else:
            frame = self.gdf.copy()

        # Identify junk fixes based on location coordinate x,y ranges or that match specific coordinates
        if isinstance(fix_filter, RelocsCoordinateFilter):
            frame.loc[
                (frame["geometry"].x < fix_filter.min_x)
                | (frame["geometry"].x > fix_filter.max_x)
                | (frame["geometry"].y < fix_filter.min_y)
                | (frame["geometry"].y > fix_filter.max_y)
                | (frame["geometry"].isin(fix_filter.filter_point_coords)),
                "junk_status",
            ] = True

        # Mark fixes outside this date range as junk
        elif isinstance(fix_filter, RelocsDateRangeFilter):
            if fix_filter.start is not None:
                frame.loc[frame["fixtime"] < fix_filter.start, "junk_status"] = True

            if fix_filter.end is not None:
                frame.loc[frame["fixtime"] > fix_filter.end, "junk_status"] = True

        else:
            crs = frame.crs
            frame.to_crs(4326)
            if isinstance(fix_filter, RelocsSpeedFilter):
                frame._update_inplace(
                    frame.groupby("groupby_col")[frame.columns]
                    .apply(self._apply_speedfilter, fix_filter=fix_filter)
                    .droplevel(["groupby_col"])
                )
            elif isinstance(fix_filter, RelocsDistFilter):
                frame._update_inplace(
                    frame.groupby("groupby_col")[frame.columns]
                    .apply(self._apply_distfilter, fix_filter=fix_filter)
                    .droplevel(["groupby_col"])
                )
            frame.to_crs(crs, inplace=True)

        if not inplace:
            return frame

    @cached_property
    def distance_from_centroid(self):
        # calculate the distance between the centroid and the fix
        gs = self.geometry.to_crs(crs=self.estimate_utm_crs())
        return gs.distance(gs.unary_union.centroid)

    @cached_property
    def cluster_radius(self):
        """
        The cluster radius is the largest distance between a point in the relocationss and the
        centroid of the relocationss
        """
        distance = self.distance_from_centroid
        return distance.max()

    @cached_property
    def cluster_std_dev(self):
        """
        The cluster standard deviation is the standard deviation of the radii from the centroid
        to each point making up the cluster
        """
        distance = self.distance_from_centroid
        return np.std(distance)

    def threshold_point_count(self, threshold_dist):
        """Counts the number of points in the cluster that are within a threshold distance of the geographic centre"""
        distance = self.distance_from_centroid
        return distance[distance <= threshold_dist].size

    def apply_threshold_filter(self, threshold_dist_meters=float("Inf")):
        # Apply filter to the underlying geodataframe.
        distance = self.distance_from_centroid
        _filter = distance > threshold_dist_meters
        self.relocations.loc[_filter, "junk_status"] = True
