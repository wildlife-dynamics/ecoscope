import collections
import dataclasses
import typing

import geopandas as gpd
import pandas as pd
import pygeos

import ecoscope


class Region(collections.UserDict):
    def __init__(self, geometry: typing.Any, unique_id: str = None, region_name: str = None):
        super().__init__(
            geometry=geometry,
            unique_id=unique_id,
            region_name=region_name,
        )


class GeoFence(collections.UserDict):
    """Represents Virtual Fence Boundary"""

    def __init__(
        self,
        geometry: typing.Any,
        unique_id: str = None,
        fence_name: str = None,
        warn_level: str = None,
    ):
        super().__init__(
            geometry=geometry,
            unique_id=unique_id,
            fence_name=fence_name,
            warn_level=warn_level,
        )


@dataclasses.dataclass
class GeoCrossingProfile:
    geofences: typing.List[GeoFence]
    regions: typing.List[Region]

    @property
    def geofence_df(self):
        return gpd.GeoDataFrame(self.geofences, crs=4326)

    @property
    def region_df(self):
        return gpd.GeoDataFrame(self.regions, crs=4326)


class GeoFenceCrossing:
    @classmethod
    def analyse(
        cls,
        geocrossing_profile: GeoCrossingProfile,
        trajectory: ecoscope.base.Trajectory,
    ):
        """
        Analyze the trajectory of each subject in relation to set of virtual fences and regions to determine where/when
        the polylines were crossed and what the containment of the individual was before and
        after any geofence crossings.

        Parameters
        ----------
        geocrossing_profile: GeoCrossingProfile
            Object that contains the geonfences and regions
        trajectory: ecoscope.base.Trajectory
            Geodataframe stores goemetry, speed_kmhr, heading etc. for each subject.

        Returns
        -------
            ecoscope.base.EcoDataFrame

        """
        trajectory = trajectory.copy()
        trajectory["start_point"] = pygeos.get_point(trajectory.geometry.values.data, 0)
        trajectory["end_point"] = pygeos.get_point(trajectory.geometry.values.data, 1)

        def apply_func(fence):
            geofence = fence.geometry
            traj = trajectory.loc[trajectory.intersects(geofence)].copy()
            traj["segment_geometry"] = traj["geometry"]
            traj.set_geometry(traj.intersection(geofence), inplace=True)
            traj = traj.explode(index_parts=False)
            traj = traj.loc[traj["geometry"].type == "Point"]

            dist = traj.segment_geometry.project(traj.geometry, normalized=True)
            traj["crossing_time"] = dist * (traj.segment_end - traj.segment_start) + traj.segment_start

            traj["unique_id"] = fence.unique_id
            traj["fence_name"] = fence.fence_name
            traj["warn_level"] = fence.warn_level

            # Determine containment of the subject before and after the crossing
            for index, colname in enumerate(["start_region_ids", "end_region_ids"]):
                traj[colname] = (
                    geocrossing_profile.region_df.sjoin(
                        gpd.GeoDataFrame(
                            geometry=pygeos.get_point(traj["segment_geometry"].values.data, index),
                            index=traj.index,
                            crs=4326,
                        ),
                        predicate="contains",
                    )
                    .set_index("index_right")["unique_id"]
                    .groupby(level=0)
                    .agg(list)
                )
            return traj

        fences = geocrossing_profile.geofence_df
        df = pd.concat([apply_func(fence) for _, fence in fences.iterrows()])
        df.drop(["start_point", "end_point"], axis=1, inplace=True)
        return ecoscope.base.EcoDataFrame(df, geometry="geometry")
