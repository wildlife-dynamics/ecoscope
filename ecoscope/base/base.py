import warnings
from functools import cached_property

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


class EcoDataFrame(gpd.GeoDataFrame):
    """
    `EcoDataFrame` extends `geopandas.GeoDataFrame` to provide customizations and allow for simpler extension.
    """

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, *args, **kwargs):
        if kwargs.get("geometry") is None:
            # Load geometry from data if not specified in kwargs
            if hasattr(data, "geometry"):
                kwargs["geometry"] = data.geometry.name

        if kwargs.get("crs") is None:
            # Load crs from data if not specified in kwargs
            if hasattr(data, "crs"):
                kwargs["crs"] = data.crs

        super().__init__(data, *args, **kwargs)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, (list, slice, np.ndarray, pd.Series)):
            result.__class__ = self._constructor
        return result

    @classmethod
    def from_file(cls, filename, **kwargs):
        result = gpd.GeoDataFrame.from_file(filename, **kwargs)
        result.__class__ = cls
        return result

    @classmethod
    def from_features(cls, features, **kwargs):
        result = gpd.GeoDataFrame.from_features(features, **kwargs)
        result.__class__ = cls
        return result

    def __finalize__(self, *args, **kwargs):
        result = super().__finalize__(*args, **kwargs)
        result.__class__ = self._constructor
        return result

    def astype(self, *args, **kwargs):
        result = super().astype(*args, **kwargs)
        result.__class__ = self._constructor
        return result

    def merge(self, *args, **kwargs):
        result = super().merge(*args, **kwargs)
        result.__class__ = self._constructor
        return result

    def dissolve(self, *args, **kwargs):
        result = super().dissolve(*args, **kwargs)
        result.__class__ = self._constructor
        return result

    def explode(self, *args, **kwargs):
        result = super().explode(*args, **kwargs)
        result.__class__ = self._constructor
        return result

    def plot(self, *args, **kwargs):
        if self._geometry_column_name in self:
            return gpd.GeoDataFrame.plot(self, *args, **kwargs)
        else:
            return pd.DataFrame(self).plot(*args, **kwargs)

    def reset_filter(self, inplace=False):
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame["junk_status"] = False

        if not inplace:
            return frame

    def remove_filtered(self, inplace=False):
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if not frame["junk_status"].dtype == bool:
            warnings.warn(
                f"junk_status column is of type {frame['junk_status'].dtype}, expected `bool`. "
                "Attempting to automatically convert."
            )
            frame["junk_status"] = frame["junk_status"].astype(bool)

        frame.query("~junk_status", inplace=True)

        if not inplace:
            return frame


class Relocations(EcoDataFrame):
    """
    Relocation is a model for a set of fixes from a given subject.
    Because fixes are temporal, they can be ordered asc or desc. The additional_data dict can contain info
    specific to the subject and relocations: name, type, region, sex etc. These values are applicable to all
    fixes in the relocations array. If they vary, then they should be put into each fix's additional_data dict.
    """

    @classmethod
    def from_gdf(cls, gdf, groupby_col=None, time_col="fixtime", uuid_col=None, **kwargs):
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

        assert {"geometry", time_col}.issubset(gdf)

        if kwargs.get("copy") is not False:
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

        return cls(gdf, **kwargs)

    @staticmethod
    def _apply_speedfilter(df, fix_filter):
        with warnings.catch_warnings():
            """
            Note : This warning can be removed once the version of Geopandas is updated
            on Colab to the one that fixes this bug
            """
            warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs")
            gdf = df.assign(
                _fixtime=df["fixtime"].shift(-1),
                _geometry=df["geometry"].shift(-1),
                _junk_status=df["junk_status"].shift(-1),
            )[:-1]

        straight_track = Trajectory._straighttrack_properties(gdf)
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
    def _apply_distfilter(df, fix_filter):
        with warnings.catch_warnings():
            """
            Note : This warning can be removed once the version of Geopandas is updated
            on Colab to the one that fixes this bug
            """
            warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs")
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

    def apply_reloc_filter(self, fix_filter=None, inplace=False):
        """Apply a given filter by marking the fix junk_status based on the conditions of a filter"""

        if not self["fixtime"].is_monotonic_increasing:
            self.sort_values("fixtime", inplace=True)
        assert self["fixtime"].is_monotonic_increasing

        if inplace:
            frame = self
        else:
            frame = self.copy()

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


class Trajectory(EcoDataFrame):
    """
    A trajectory represents a time-ordered collection of segments.
    Currently only straight track segments exist.
    It is based on an underlying relocs object that is the point representation
    """

    @classmethod
    def from_relocations(cls, gdf, *args, **kwargs):
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
        assert isinstance(gdf, Relocations)
        assert {"groupby_col", "fixtime", "geometry"}.issubset(gdf)

        if kwargs.get("copy"):
            gdf = gdf.copy()

        gdf = EcoDataFrame(gdf)
        crs = gdf.crs
        gdf.to_crs(4326, inplace=True)
        gdf = gdf.groupby("groupby_col")[gdf.columns].apply(cls._create_multitraj).droplevel(level=0)
        gdf.to_crs(crs, inplace=True)
        gdf.sort_values("segment_start", inplace=True)
        return cls(gdf, *args, **kwargs)

    def get_displacement(self):
        """
        Get displacement in meters between first and final fixes.
        """
        if not self["segment_start"].is_monotonic_increasing:
            self = self.sort_values("segment_start")
        start = self.geometry.iloc[0].coords[0]
        end = self.geometry.iloc[-1].coords[1]
        return Geod(ellps="WGS84").inv(start[0], start[1], end[0], end[1])[2]

    def get_tortuosity(self):
        """
        Get tortuosity for dataframe defined as distance traveled divided by displacement between first and final
        points.
        """

        return self["dist_meters"].sum() / self.get_displacement()

    @staticmethod
    def _create_multitraj(df):
        if len(df) == 1:
            warnings.warn(
                f"Subject id {df.get('groupby_col')} has only one relocation "
                "and will be excluded from trajectory creation"
            )
            return None
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
        track_properties = Trajectory._straighttrack_properties(gdf)

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
        if not self["segment_start"].is_monotonic_increasing:
            self.sort_values("segment_start", inplace=True)
        assert self["segment_start"].is_monotonic_increasing

        if inplace:
            frame = self
        else:
            frame = self.copy()

        assert type(traj_seg_filter) is TrajSegFilter
        frame.loc[
            (frame["dist_meters"] < traj_seg_filter.min_length_meters)
            | (frame["dist_meters"] > traj_seg_filter.max_length_meters)
            | (frame["timespan_seconds"] < traj_seg_filter.min_time_secs)
            | (frame["timespan_seconds"] > traj_seg_filter.max_time_secs)
            | (frame["speed_kmhr"] < traj_seg_filter.min_speed_kmhr)
            | (frame["speed_kmhr"] > traj_seg_filter.max_speed_kmhr),
            "junk_status",
        ] = True

        if not inplace:
            return frame

    def get_turn_angle(self):
        if not self["segment_start"].is_monotonic_increasing:
            self.sort_values("segment_start", inplace=True)
        assert self["segment_start"].is_monotonic_increasing

        def turn_angle(traj):
            return ((traj["heading"].diff() + 540) % 360 - 180)[traj["segment_end"].shift(1) == traj["segment_start"]]

        uniq = self.groupby_col.nunique()
        angles = (
            self.groupby("groupby_col")[self.columns].apply(turn_angle, include_groups=False).droplevel(0)
            if uniq > 1
            else turn_angle(self)
        )

        return angles.rename("turn_angle").reindex(self.index)

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

        if not self["segment_start"].is_monotonic_increasing:
            self.sort_values("segment_start", inplace=True)

        def f(traj):
            traj.crs = self.crs  # Lost in groupby-apply due to GeoPandas bug

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

        return Relocations.from_gdf(
            self.groupby("groupby_col")[self.columns].apply(f, include_groups=False).reset_index(level=0)
        )

    def to_relocations(self):
        """
        Converts a Trajectory object to a Relocations object.
        Returns
        -------
        ecoscope.base.Relocations
        """

        def f(traj):
            traj.crs = self.crs
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

        return Relocations.from_gdf(
            self.groupby("groupby_col")[self.columns].apply(f, include_groups=False).reset_index(drop=True)
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
                relocs_ind.crs = self.crs
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
            return relocs.groupby("groupby_col")[relocs.columns].apply(f).reset_index(drop=True)

    @staticmethod
    def _straighttrack_properties(df: gpd.GeoDataFrame):
        """Private function used by Trajectory class."""

        class Properties:
            @property
            def start_fixes(self):
                # unpack xy-coordinates of start fixes
                return df["geometry"].x, df["geometry"].y

            @property
            def end_fixes(self):
                # unpack xy-coordinates of end fixes
                return df["_geometry"].x, df["_geometry"].y

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
                start_point = df["geometry"].iloc[0]
                geod = Geod(ellps="WGS84")
                geod_displacement = [geod.inv(start_point.x, start_point.y, geo.x, geo.y)[2] for geo in df["_geometry"]]
                return [(x**2) / (1000 * 2) for x in geod_displacement]

            @property
            def timespan_seconds(self):
                return (df["_fixtime"] - df["fixtime"]).dt.total_seconds()

            @property
            def speed_kmhr(self):
                return (self.dist_meters / self.timespan_seconds) * 3.6

        instance = Properties()
        return instance
