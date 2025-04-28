import warnings
from copy import deepcopy
from functools import cached_property

import numpy as np
import pandas as pd
import geopandas as gpd  # type: ignore[import-untyped]
from pyproj import Geod

from ecoscope.base._dataclasses import (
    RelocsCoordinateFilter,
    RelocsDateRangeFilter,
    RelocsDistFilter,
    RelocsSpeedFilter,
    RelocsFilterType,
)
from ecoscope.base.straightrack import StraightTrackProperties
from ecoscope.base import EcoDataFrame


class Relocations(EcoDataFrame):
    """
    Relocation is a model for a set of fixes from a given subject.
    Because fixes are temporal, they can be ordered asc or desc. The additional_data dict can contain info
    specific to the subject and relocations: name, type, region, sex etc. These values are applicable to all
    fixes in the relocations array. If they vary, then they should be put into each fix's additional_data dict.
    """

    @classmethod
    def from_gdf(
        cls,
        gdf: gpd.GeoDataFrame,
        groupby_col: str | None = None,
        time_col: str = "fixtime",
        uuid_col: str | None = None,
        copy: bool = True,
    ):
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
        copy : bool, optional
            Whether or not to copy the `gdf`. Defaults to `True`.
        """

        assert {"geometry", time_col}.issubset(gdf)

        if copy:
            gdf = gdf.copy()

        if groupby_col is None:
            if "groupby_col" not in gdf:
                gdf["groupby_col"] = 0
        else:
            gdf["groupby_col"] = gdf.loc[:, groupby_col]

        if time_col != "fixtime":
            gdf["fixtime"] = gdf.loc[:, time_col]

        if not pd.api.types.is_datetime64_any_dtype(gdf["fixtime"]):
            warnings.warn(
                f"{time_col} is not of type datetime64. Attempting to automatically infer format and timezone. "
                "Results may be incorrect."
            )
            gdf["fixtime"] = pd.to_datetime(gdf["fixtime"])

        if gdf["fixtime"].dt.tz is None:
            warnings.warn(f"{time_col} is not timezone aware. Assuming datetime are in UTC.")
            gdf["fixtime"] = gdf["fixtime"].dt.tz_localize(tz="UTC")

        if gdf.crs is None:
            warnings.warn("CRS was not set. Assuming geometries are in WGS84.")
            gdf.set_crs(4326, inplace=True)

        if uuid_col is not None:
            gdf.set_index(uuid_col, drop=False, inplace=True)

        gdf["junk_status"] = False

        default_cols = ["groupby_col", "fixtime", "junk_status", "geometry"]
        extra_cols = gdf.columns.difference(default_cols)
        extra_cols = extra_cols[~extra_cols.str.startswith("extra__")]

        assert gdf.columns.intersection("extra__" + extra_cols).empty, "Column names overlap with existing `extra`"

        gdf.rename(columns=dict(zip(extra_cols, "extra__" + extra_cols)), inplace=True)

        return cls(gdf=gdf)

    @staticmethod
    def _apply_speedfilter(df: pd.DataFrame, fix_filter: RelocsSpeedFilter):
        gdf = df.assign(
            _fixtime=df["fixtime"].shift(-1),
            _geometry=df["geometry"].shift(-1),
            _junk_status=df["junk_status"].shift(-1),
        )[:-1]

        straight_track = StraightTrackProperties(gdf)
        gdf["speed_kmhr"] = straight_track.speed_kmhr

        gdf.loc[
            (~gdf["junk_status"]) & (~gdf["_junk_status"]) & (gdf["speed_kmhr"] > fix_filter.max_speed_kmhr),
            "junk_status",
        ] = True

        gdf.drop(
            ["_fixtime", "_geometry", "_junk_status", "speed_kmhr"],
            axis=1,
            inplace=True,
        )
        return gdf

    @staticmethod
    def _apply_distfilter(df: pd.DataFrame, fix_filter: RelocsDistFilter):
        gdf = df.assign(
            _junk_status=df["junk_status"].shift(-1),
            _geometry=df["geometry"].shift(-1),
        )[:-1]

        _, _, distance_m = Geod(ellps="WGS84").inv(
            gdf["geometry"].x, gdf["geometry"].y, gdf["_geometry"].x, gdf["_geometry"].y
        )
        gdf["distance_km"] = distance_m / 1000

        gdf.loc[
            (~gdf["junk_status"]) & (~gdf["_junk_status"]) & (gdf["distance_km"] < fix_filter.min_dist_km)
            | (gdf["distance_km"] > fix_filter.max_dist_km),
            "junk_status",
        ] = True

        gdf.drop(["_geometry", "_junk_status", "distance_km"], axis=1, inplace=True)
        return gdf

    def apply_reloc_filter(self, fix_filter: RelocsFilterType | None = None, inplace: bool = False):
        """Apply a given filter by marking the fix junk_status based on the conditions of a filter"""

        if not self.gdf["fixtime"].is_monotonic_increasing:
            self.gdf.sort_values("fixtime", inplace=True)
        assert self.gdf["fixtime"].is_monotonic_increasing

        if inplace:
            relocs = self
        else:
            relocs = deepcopy(self)

        # Identify junk fixes based on location coordinate x,y ranges or that match specific coordinates
        if isinstance(fix_filter, RelocsCoordinateFilter):
            relocs.gdf.loc[
                (relocs.gdf["geometry"].x < fix_filter.min_x)
                | (relocs.gdf["geometry"].x > fix_filter.max_x)
                | (relocs.gdf["geometry"].y < fix_filter.min_y)
                | (relocs.gdf["geometry"].y > fix_filter.max_y)
                | (relocs.gdf["geometry"].isin(fix_filter.filter_point_coords)),
                "junk_status",
            ] = True

        # Mark fixes outside this date range as junk
        elif isinstance(fix_filter, RelocsDateRangeFilter):
            if fix_filter.start is not None:
                relocs.gdf.loc[relocs.gdf["fixtime"] < fix_filter.start, "junk_status"] = True

            if fix_filter.end is not None:
                relocs.gdf.loc[relocs.gdf["fixtime"] > fix_filter.end, "junk_status"] = True

        else:
            crs = relocs.gdf.crs
            relocs.gdf.to_crs(4326)
            if isinstance(fix_filter, RelocsSpeedFilter):
                relocs.gdf._update_inplace(
                    relocs.gdf.groupby("groupby_col")[relocs.gdf.columns]
                    .apply(self._apply_speedfilter, fix_filter=fix_filter)
                    .droplevel(["groupby_col"])
                )
            elif isinstance(fix_filter, RelocsDistFilter):
                relocs.gdf._update_inplace(
                    relocs.gdf.groupby("groupby_col")[relocs.gdf.columns]
                    .apply(self._apply_distfilter, fix_filter=fix_filter)
                    .droplevel(["groupby_col"])
                )
            relocs.gdf.to_crs(crs, inplace=True)

        if not inplace:
            return relocs

    @cached_property
    def distance_from_centroid(self):
        # calculate the distance between the centroid and the fix
        gs = self.gdf.geometry.to_crs(crs=self.gdf.estimate_utm_crs())
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

    def threshold_point_count(self, threshold_dist: float):
        """Counts the number of points in the cluster that are within a threshold distance of the geographic centre"""
        distance = self.distance_from_centroid
        return distance[distance <= threshold_dist].size

    def apply_threshold_filter(self, threshold_dist_meters: float = float("Inf")):
        # Apply filter to the underlying geodataframe.
        distance = self.distance_from_centroid
        _filter = distance > threshold_dist_meters
        self.gdf.loc[_filter, "junk_status"] = True
