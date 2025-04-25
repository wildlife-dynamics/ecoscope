import itertools
import warnings
from copy import deepcopy

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod

from ecoscope.base._dataclasses import (
    ProximityProfile,
    TrajSegFilter,
)
from ecoscope.base.straightrack import StraightTrackProperties
from ecoscope.base import EcoDataFrame, Relocations


def get_displacement(gdf: gpd.GeoDataFrame):
    """
    Get displacement in meters between first and final fixes.
    """
    if not gdf["segment_start"].is_monotonic_increasing:
        gdf = gdf.sort_values("segment_start")
    start = gdf.geometry.iloc[0].coords[0]
    end = gdf.geometry.iloc[-1].coords[1]
    return Geod(ellps="WGS84").inv(start[0], start[1], end[0], end[1])[2]


def get_tortuosity(gdf: gpd.GeoDataFrame):
    """
    Get tortuosity for dataframe defined as distance traveled divided by displacement between first and final
    points.
    """

    return gdf["dist_meters"].sum() / get_displacement(gdf)


class Trajectory(EcoDataFrame):
    """
    A trajectory represents a time-ordered collection of segments.
    Currently only straight track segments exist.
    It is based on an underlying relocs object that is the point representation
    """

    @classmethod
    def from_relocations(cls, relocs: Relocations, *args, **kwargs):
        """
        Create Trajectory class from Relocation dataframe.
        Parameters
        ----------
        relocs: Relocations
            A `Relocations` instance.
        args
        kwargs
        Returns
        -------
        Trajectory
        """
        assert isinstance(relocs, Relocations)
        assert {"groupby_col", "fixtime", "geometry"}.issubset(relocs.gdf)

        if kwargs.get("copy"):
            relocs = Relocations(relocs.gdf.copy())

        original_crs = relocs.gdf.crs
        relocs.gdf.to_crs(4326, inplace=True)
        relocs.gdf = (
            relocs.gdf.groupby("groupby_col")[relocs.gdf.columns].apply(cls._create_multitraj).droplevel(level=0)
        )
        relocs.gdf.to_crs(original_crs, inplace=True)

        relocs.gdf.sort_values("segment_start", inplace=True)
        return Trajectory(gdf=relocs.gdf)

    @staticmethod
    def get_displacement(gdf: gpd.GeoDataFrame):
        """
        Get displacement in meters between first and final fixes.
        """
        if not gdf["segment_start"].is_monotonic_increasing:
            gdf = gdf.sort_values("segment_start")
        start = gdf.geometry.iloc[0].coords[0]
        end = gdf.geometry.iloc[-1].coords[1]
        return Geod(ellps="WGS84").inv(start[0], start[1], end[0], end[1])[2]

    @staticmethod
    def get_tortuosity(gdf: gpd.GeoDataFrame):
        """
        Get tortuosity for dataframe defined as distance traveled divided by displacement between first and final
        points.
        """

        return gdf["dist_meters"].sum() / get_displacement(gdf)

    @staticmethod
    def _create_multitraj(df):
        if len(df) == 1:
            warnings.warn(
                f"Subject id {df.get('groupby_col')} has only one relocation "
                "and will be excluded from trajectory creation"
            )
            return None

        df["_geometry"] = df["geometry"].shift(-1)

        df["_fixtime"] = df["fixtime"].shift(-1)
        return Trajectory._create_trajsegments(df[:-1])

    @staticmethod
    def _create_trajsegments(gdf):
        track_properties = StraightTrackProperties(gdf)

        coords = np.column_stack(
            (
                np.column_stack(track_properties.start_fixes),
                np.column_stack(track_properties.end_fixes),
            )
        ).reshape(gdf.shape[0], 2, 2)

        df = gpd.GeoDataFrame(
            {
                "groupby_col": gdf.groupby_col,
                "segment_start": gdf.fixtime,
                "segment_end": gdf._fixtime,
                "timespan_seconds": track_properties.timespan_seconds,
                "dist_meters": track_properties.dist_meters,
                "speed_kmhr": track_properties.speed_kmhr,
                "heading": track_properties.heading,
                "geometry": shapely.linestrings(coords),
                "junk_status": gdf.junk_status,
                "nsd": track_properties.nsd,
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
        if not self.gdf["segment_start"].is_monotonic_increasing:
            self.gdf.sort_values("segment_start", inplace=True)
        assert self.gdf["segment_start"].is_monotonic_increasing

        if inplace:
            traj = self
        else:
            traj = deepcopy(self)

        assert type(traj_seg_filter) is TrajSegFilter
        traj.gdf.loc[
            (traj.gdf["dist_meters"] < traj_seg_filter.min_length_meters)
            | (traj.gdf["dist_meters"] > traj_seg_filter.max_length_meters)
            | (traj.gdf["timespan_seconds"] < traj_seg_filter.min_time_secs)
            | (traj.gdf["timespan_seconds"] > traj_seg_filter.max_time_secs)
            | (traj.gdf["speed_kmhr"] < traj_seg_filter.min_speed_kmhr)
            | (traj.gdf["speed_kmhr"] > traj_seg_filter.max_speed_kmhr),
            "junk_status",
        ] = True

        if not inplace:
            return traj

    def get_turn_angle(self):
        if not self.gdf["segment_start"].is_monotonic_increasing:
            self.gdf.sort_values("segment_start", inplace=True)
        assert self.gdf["segment_start"].is_monotonic_increasing

        def turn_angle(traj):
            return ((traj["heading"].diff() + 540) % 360 - 180)[traj["segment_end"].shift(1) == traj["segment_start"]]

        uniq = self.gdf.groupby_col.nunique()
        angles = (
            self.gdf.groupby("groupby_col")[self.gdf.columns].apply(turn_angle, include_groups=False).droplevel(0)
            if uniq > 1
            else turn_angle(self.gdf)
        )

        return angles.rename("turn_angle").reindex(self.gdf.index)

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

        if not self.gdf["segment_start"].is_monotonic_increasing:
            self.gdf.sort_values("segment_start", inplace=True)

        def f(gdf):
            assert gdf.crs

            times = pd.date_range(gdf["segment_start"].iat[0], gdf["segment_end"].iat[-1], freq=freq)

            start_i = gdf["segment_start"].searchsorted(times, side="right") - 1
            end_i = gdf["segment_end"].searchsorted(times, side="left")
            valid_i = (start_i == end_i) | (times == gdf["segment_start"].iloc[start_i])

            gdf = gdf.iloc[start_i[valid_i]].reset_index(drop=True)
            times = times[valid_i]

            return gpd.GeoDataFrame(
                {"fixtime": times},
                geometry=shapely.line_interpolate_point(
                    gdf["geometry"].values,
                    (times - gdf["segment_start"]) / (gdf["segment_end"] - gdf["segment_start"]),
                    normalized=True,
                ),
                crs=gdf.crs,
            )

        return Relocations.from_gdf(
            self.gdf.groupby("groupby_col")[self.gdf.columns].apply(f, include_groups=False).reset_index(level=0)
        )

    def to_relocations(self):
        """
        Converts a Trajectory object to a Relocations object.
        Returns
        -------
        ecoscope.base.Relocations
        """

        def f(gdf):
            assert gdf.crs
            points = np.concatenate([shapely.get_point(gdf.geometry, 0), shapely.get_point(gdf.geometry, 1)])
            times = np.concatenate([gdf["segment_start"], gdf["segment_end"]])

            return (
                gpd.GeoDataFrame(
                    {"fixtime": times},
                    geometry=points,
                    crs=gdf.crs,
                )
                .drop_duplicates(subset=["fixtime"])
                .sort_values("fixtime")
            )

        return Relocations.from_gdf(
            self.gdf.groupby("groupby_col")[self.gdf.columns].apply(f, include_groups=False).reset_index(drop=True)
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
            return self.upsample(freq)
        else:
            freq = pd.tseries.frequencies.to_offset(freq)
            tolerance = pd.tseries.frequencies.to_offset(tolerance)

            def f(relocs_ind):
                relocs_ind.crs = self.gdf.crs
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

            relocs = self.to_relocations()
            relocs.gdf = relocs.gdf.groupby("groupby_col")[relocs.gdf.columns].apply(f).reset_index(drop=True)
            return relocs

    @staticmethod
    def _straighttrack_properties(gdf: gpd.GeoDataFrame):
        """Private function used by Trajectory class."""

        class Properties:
            @property
            def start_fixes(self):
                # unpack xy-coordinates of start fixes
                return gdf["geometry"].x, gdf["geometry"].y

            @property
            def end_fixes(self):
                # unpack xy-coordinates of end fixes
                return gdf["_geometry"].x, gdf["_geometry"].y

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
            def nsd(self):
                start_point = gdf["geometry"].iloc[0]
                geod = Geod(ellps="WGS84")
                geod_displacement = [
                    geod.inv(start_point.x, start_point.y, geo.x, geo.y)[2] for geo in gdf["_geometry"]
                ]
                return [(x**2) / (1000 * 2) for x in geod_displacement]

            @property
            def timespan_seconds(self):
                return (gdf["_fixtime"] - gdf["fixtime"]).dt.total_seconds()

            @property
            def speed_kmhr(self):
                return (self.dist_meters / self.timespan_seconds) * 3.6

        instance = Properties()
        return instance

    def calculate_proximity(
        self,
        proximity_profile: ProximityProfile,
    ) -> pd.DataFrame:
        """
        A function to analyze the trajectory of a subject in relation to a set of spatial features and regions to
        determine where/when the subject was proximal to the spatial feature.

        Parameters
        ----------
        proximity_profile: ProximityProfile
            proximity setting for performing calculation
        Returns
        -------
        pd.DataFrame

        """
        proximity_events = []

        def analysis(gdf):
            for sf in proximity_profile.spatial_features:
                proximity_dist = gdf.geometry.distance(sf.geometry)
                start_fix = gpd.GeoSeries([shapely.Point(g.coords[0]) for g in gdf.geometry])

                pr = gdf[["groupby_col", "speed_kmhr", "heading"]]
                pr["proximity_distance"] = proximity_dist
                pr["proximal_fix"] = start_fix  # TODO: figure out the estimated fix interpolated along the seg
                pr["estimated_time"] = gdf.segment_start
                pr["geometry"] = gdf.geometry
                pr["spatialfeature_id"] = list(itertools.repeat(sf.unique_id, pr.shape[0]))
                pr["spatialfeature_name"] = list(itertools.repeat(sf.name, pr.shape[0]))

                proximity_events.append(pr)

        self.gdf.groupby("groupby_col")[self.gdf.columns].apply(analysis, include_groups=False)
        return pd.concat(proximity_events).reset_index(drop=True)
