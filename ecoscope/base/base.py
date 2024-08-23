import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod

from ecoscope.base._dataclasses import (
    RelocsCoordinateFilter,
    RelocsDateRangeFilter,
    RelocsDistFilter,
    RelocsSpeedFilter,
    TrajSegFilter,
)
from functools import cached_property


class StraighttrackMixin:
    @property
    def start_fixes(self):
        # unpack xy-coordinates of start fixes
        return self._gdf["geometry"].x, self._gdf["geometry"].y

    @property
    def end_fixes(self):
        # unpack xy-coordinates of end fixes
        return self._gdf["_geometry"].x, self._gdf["_geometry"].y

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
        return (self._gdf["_fixtime"] - self._gdf["fixtime"]).dt.total_seconds()

    @property
    def speed_kmhr(self):
        return (self.dist_meters / self.timespan_seconds) * 3.6


@pd.api.extensions.register_dataframe_accessor("relocations")
class Relocations(StraighttrackMixin):
    """
    Relocation is a model for a set of fixes from a given subject.
    Because fixes are temporal, they can be ordered asc or desc. The additional_data dict can contain info
    specific to the subject and relocations: name, type, region, sex etc. These values are applicable to all
    fixes in the relocations array. If they vary, then they should be put into each fix's additional_data dict.
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._gdf = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj, gpd.GeoDataFrame)

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
            if "groupby_col" not in self._gdf:
                self._gdf["groupby_col"] = 0
        else:
            self._gdf["groupby_col"] = self._gdf.loc[:, groupby_col]

        if time_col != "fixtime":
            self._gdf["fixtime"] = self._gdf.loc[:, time_col]

        if not pd.api.types.is_datetime64_any_dtype(self._gdf["fixtime"]):
            warnings.warn(
                f"{time_col} is not of type datetime64. Attempting to automatically infer format and timezone. "
                "Results may be incorrect."
            )
            self._gdf["fixtime"] = pd.to_datetime(self._gdf["fixtime"])

        if self._gdf["fixtime"].dt.tz is None:
            warnings.warn(f"{time_col} is not timezone aware. Assuming datetime are in UTC.")
            self._gdf["fixtime"] = self._gdf["fixtime"].dt.tz_localize(tz="UTC")

        if self._gdf.crs is None:
            warnings.warn("CRS was not set. Assuming geometries are in WGS84.")
            self._gdf.set_crs(4326, inplace=True)

        if uuid_col is not None:
            self._gdf.set_index(uuid_col, drop=False, inplace=True)

        self._gdf["junk_status"] = False

        default_cols = ["groupby_col", "fixtime", "junk_status", "geometry"]
        extra_cols = self._gdf.columns.difference(default_cols)
        extra_cols = extra_cols[~extra_cols.str.startswith("extra__")]

        assert self._gdf.columns.intersection(
            "extra__" + extra_cols
        ).empty, "Column names overlap with existing `extra`"

        self._gdf.rename(columns=dict(zip(extra_cols, "extra__" + extra_cols)), inplace=True)
        return self._gdf

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

    def apply_reloc_filter(self, fix_filter=None):
        """Apply a given filter by marking the fix junk_status based on the conditions of a filter"""

        if not self._gdf["fixtime"].is_monotonic_increasing:
            self._gdf.sort_values("fixtime", inplace=True)
        assert self._gdf["fixtime"].is_monotonic_increasing

        # Identify junk fixes based on location coordinate x,y ranges or that match specific coordinates
        if isinstance(fix_filter, RelocsCoordinateFilter):
            self._gdf.loc[
                (self._gdf["geometry"].x < fix_filter.min_x)
                | (self._gdf["geometry"].x > fix_filter.max_x)
                | (self._gdf["geometry"].y < fix_filter.min_y)
                | (self._gdf["geometry"].y > fix_filter.max_y)
                | (self._gdf["geometry"].isin(fix_filter.filter_point_coords)),
                "junk_status",
            ] = True

        # Mark fixes outside this date range as junk
        elif isinstance(fix_filter, RelocsDateRangeFilter):
            if fix_filter.start is not None:
                self._gdf.loc[self._gdf["fixtime"] < fix_filter.start, "junk_status"] = True

            if fix_filter.end is not None:
                self._gdf.loc[self._gdf["fixtime"] > fix_filter.end, "junk_status"] = True

        else:
            crs = self._gdf.crs
            self._gdf.to_crs(4326)
            if isinstance(fix_filter, RelocsSpeedFilter):
                self._gdf._update_inplace(
                    self._gdf.groupby("groupby_col")[self._gdf.columns]
                    .apply(self._apply_speedfilter, fix_filter=fix_filter)
                    .droplevel(["groupby_col"])
                )
            elif isinstance(fix_filter, RelocsDistFilter):
                self._gdf._update_inplace(
                    self._gdf.groupby("groupby_col")[self._gdf.columns]
                    .apply(self._apply_distfilter, fix_filter=fix_filter)
                    .droplevel(["groupby_col"])
                )
            self._gdf.to_crs(crs, inplace=True)

        return self._gdf

    @cached_property
    def distance_from_centroid(self):
        # calculate the distance between the centroid and the fix
        gs = self._gdf.geometry.to_crs(crs=self._gdf.estimate_utm_crs())
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
        self._gdf.loc[_filter, "junk_status"] = True

    def reset_filter(self):
        self._gdf["junk_status"] = False
        return self._gdf

    def remove_filtered(self):
        return self._gdf.query("~junk_status")


@pd.api.extensions.register_dataframe_accessor("trajectories")
class Trajectory(StraighttrackMixin):
    """
    A trajectory represents a time-ordered collection of segments.
    Currently only straight track segments exist.
    It is based on an underlying relocs object that is the point representation
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._gdf = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj, gpd.GeoDataFrame)

    def from_relocations(self, *args, **kwargs):
        """
        Create Trajectory class from Relocation dataframe.
        Parameters
        ----------
        gdf: Geodataframe
            Relocation geodataframe with relevant columns
        args
        kwargs
        Returns
        -------
        Trajectory
        """
        # if kwargs.get("copy") is not False:
        #     gdf = gdf.copy()

        crs = self._gdf.crs
        self._gdf.to_crs(4326, inplace=True)
        self._gdf = (
            self._gdf.groupby("groupby_col")[self._gdf.columns].apply(Trajectory._create_multitraj).droplevel(level=0)
        )
        self._gdf.to_crs(crs, inplace=True)
        self._gdf.sort_values("segment_start", inplace=True)
        return self._gdf

    @staticmethod
    def get_displacement(gdf):
        """
        Get displacement in meters between first and final fixes.
        """

        if not gdf["segment_start"].is_monotonic_increasing:
            gdf = gdf.sort_values("segment_start")

        gs = gdf.geometry.iloc[[0, -1]]
        start, end = gs.to_crs(gs.estimate_utm_crs())

        return start.distance(end)

    @staticmethod
    def get_tortuosity(gdf):
        """
        Get tortuosity for dataframe defined as distance traveled divided by displacement between first and final
        points.
        """

        return gdf["dist_meters"].sum() / Trajectory.get_displacement(gdf)

    @staticmethod
    def _create_multitraj(df):
        with warnings.catch_warnings():
            """
            Note : This warning can be removed once the version of Geopandas is updated
            on Colab to the one that fixes this bug
            """
            warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs")
            df["_geometry"] = df["geometry"].shift(-1)

        df["_fixtime"] = df["fixtime"].shift(-1)
        return Trajectory._create_trajsegments(df[:-1])

    @staticmethod
    def _create_trajsegments(gdf):
        coords = np.column_stack(
            (
                np.column_stack(gdf.trajectories.start_fixes),
                np.column_stack(gdf.trajectories.end_fixes),
            )
        ).reshape(gdf.shape[0], 2, 2)

        df = gpd.GeoDataFrame(
            {
                "groupby_col": gdf.groupby_col,
                "segment_start": gdf.fixtime,
                "segment_end": gdf._fixtime,
                "timespan_seconds": gdf.trajectories.timespan_seconds,
                "dist_meters": gdf.trajectories.dist_meters,
                "speed_kmhr": gdf.trajectories.speed_kmhr,
                "heading": gdf.trajectories.heading,
                "geometry": shapely.linestrings(coords),
                "junk_status": gdf.junk_status,
            },
            crs=4326,
            index=gdf.index,
        )
        gdf.drop(["fixtime", "_fixtime", "_geometry"], axis=1, inplace=True)
        extra_cols = gdf.columns.difference(df.columns)
        gdf = gdf[extra_cols]

        extra_cols = extra_cols[~extra_cols.str.startswith("extra_")]
        gdf.rename(columns=dict(zip(extra_cols, "extra__" + extra_cols)), inplace=True)

        return df.join(gdf, how="left")

    def apply_traj_filter(self, traj_seg_filter, inplace=False):
        if not self._gdf["segment_start"].is_monotonic_increasing:
            self._gdf.sort_values("segment_start", inplace=True)
        assert self._gdf["segment_start"].is_monotonic_increasing

        assert type(traj_seg_filter) is TrajSegFilter
        self._gdf.loc[
            (self._gdf["dist_meters"] < traj_seg_filter.min_length_meters)
            | (self._gdf["dist_meters"] > traj_seg_filter.max_length_meters)
            | (self._gdf["timespan_seconds"] < traj_seg_filter.min_time_secs)
            | (self._gdf["timespan_seconds"] > traj_seg_filter.max_time_secs)
            | (self._gdf["speed_kmhr"] < traj_seg_filter.min_speed_kmhr)
            | (self._gdf["speed_kmhr"] > traj_seg_filter.max_speed_kmhr),
            "junk_status",
        ] = True

        return self._gdf

    def get_turn_angle(self):
        if not self._gdf["segment_start"].is_monotonic_increasing:
            self._gdf.sort_values("segment_start", inplace=True)
        assert self._gdf["segment_start"].is_monotonic_increasing

        def turn_angle(traj):
            return ((traj["heading"].diff() + 540) % 360 - 180)[traj["segment_end"].shift(1) == traj["segment_start"]]

        uniq = self._gdf.groupby_col.nunique()
        angles = (
            self._gdf.groupby("groupby_col")[self._gdf.columns].apply(turn_angle, include_groups=False).droplevel(0)
            if uniq > 1
            else turn_angle(self._gdf)
        )

        return angles.rename("turn_angle").reindex(self._gdf.index)

    def upsample(self, freq):
        """
        Interpolate to create upsampled Relocations
        Parameters
        ----------
        freq : str, pd.Timedelta or pd.DateOffset
            Sampling frequency for new Relocations object
        Returns
        -------
        relocs : ecoscope.base.Relocations
        """

        freq = pd.tseries.frequencies.to_offset(freq)

        if not self._gdf["segment_start"].is_monotonic_increasing:
            self._gdf.sort_values("segment_start", inplace=True)

        def f(traj):
            traj.crs = self._gdf.crs  # Lost in groupby-apply due to GeoPandas bug

            times = pd.date_range(traj["segment_start"].iat[0], traj["segment_end"].iat[-1], freq=freq)

            start_i = traj["segment_start"].searchsorted(times, side="right") - 1
            end_i = traj["segment_end"].searchsorted(times, side="left")
            valid_i = (start_i == end_i) | (times == traj["segment_start"].iloc[start_i])

            traj = traj.iloc[start_i[valid_i]].reset_index(drop=True)
            times = times[valid_i]

            return gpd.GeoDataFrame(
                {"fixtime": times},
                geometry=shapely.line_interpolate_point(
                    traj["geometry"].values,
                    (times - traj["segment_start"]) / (traj["segment_end"] - traj["segment_start"]),
                    normalized=True,
                ),
                crs=traj.crs,
            )

        self._gdf.relocations.from_gdf(
            self._gdf.groupby("groupby_col")[self._gdf.columns].apply(f, include_groups=False).reset_index(level=0)
        )

    def to_relocations(self):
        """
        Converts a Trajectory object to a Relocations object.
        Returns
        -------
        ecoscope.base.Relocations
        """

        def f(traj):
            traj.crs = self._gdf.crs
            points = np.concatenate([shapely.get_point(traj.geometry, 0), shapely.get_point(traj.geometry, 1)])
            times = np.concatenate([traj["segment_start"], traj["segment_end"]])

            return (
                gpd.GeoDataFrame(
                    {"fixtime": times},
                    geometry=points,
                    crs=traj.crs,
                )
                .drop_duplicates(subset=["fixtime"])
                .sort_values("fixtime")
            )

        return self._gdf.relocations.from_gdf(
            self._gdf.groupby("groupby_col")[self._gdf.columns].apply(f, include_groups=False).reset_index(drop=True)
        )

    def downsample(self, freq, tolerance="0S", interpolation=False):
        """
        Function to downsample relocations.
        Parameters
        ----------
        freq: str, pd.Timedelta or pd.DateOffset
            Downsampling frequency for new Relocations object
        tolerance: str, pd.Timedelta or pd.DateOffset
            Tolerance on the downsampling frequency
        interpolation: bool, optional
            If true, interpolates locations on the whole trajectory
        Returns
        -------
        ecoscope.base.Relocations
        """

        if interpolation:
            return self._gdf.upsample(freq)
        else:
            freq = pd.tseries.frequencies.to_offset(freq)
            tolerance = pd.tseries.frequencies.to_offset(tolerance)

            def f(relocs_ind):
                relocs_ind.crs = self._gdf.crs
                fixtime = relocs_ind["fixtime"]

                k = 1
                i = 0
                n = len(relocs_ind)
                out = np.full(n, -1)
                out[i] = k
                while i < (n - 1):
                    t_min = fixtime.iloc[i] + freq - tolerance
                    t_max = fixtime.iloc[i] + freq + tolerance

                    j = i + 1

                    while (j < (n - 1)) and (fixtime.iloc[j] < t_min):
                        j += 1

                    i = j

                    if j == (n - 1):
                        break
                    elif (fixtime.iloc[j] >= t_min) and (fixtime.iloc[j] <= t_max):
                        out[j] = k
                    else:
                        k += 1
                        out[j] = k

                relocs_ind["extra__burst"] = np.array(out, dtype=np.int64)
                relocs_ind.drop(relocs_ind.loc[relocs_ind["extra__burst"] == -1].index, inplace=True)
                return relocs_ind

            relocs = self._gdf.trajectory.to_relocations()
            return relocs.groupby("groupby_col")[relocs.columns].apply(f).reset_index(drop=True)
