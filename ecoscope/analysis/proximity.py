import itertools
import typing
import uuid
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


@dataclass
class SpatialFeature:
    """
    A spatial geometry with an associated name and unique ID. Becomes a useful construct in several movdata calculations
    """

    name: str = ""
    unique_id: typing.Any = uuid.uuid4()
    geometry: typing.Any = None


@dataclass
class ProximityProfile:
    spatial_features: typing.List[SpatialFeature] = field(default=list)


class Proximity:
    @classmethod
    def calculate_proximity(cls, proximity_profile, trajectory):
        """
        A function to analyze the trajectory of a subject in relation to a set of spatial features and regions to
        determine where/when the subject was proximal to the spatial feature.

        Parameters
        ----------
        proximity_profile: ProximityProfile
            proximity setting for performing calculation
        trajectory: ecoscope.base.Trajectory
            Geodataframe stores goemetry, speed_kmhr, heading etc. for each subject.
        Returns
        -------
        pd.DataFrame

        """
        proximity_events = []

        def analysis(traj):
            for sf in proximity_profile.spatial_features:
                proximity_dist = traj.geometry.distance(sf.geometry)
                start_fix = gpd.GeoSeries([Point(g.coords[0]) for g in traj.geometry])

                pr = traj[["groupby_col", "speed_kmhr", "heading"]]
                pr["proximity_distance"] = proximity_dist
                pr["proximal_fix"] = start_fix  # TODO: figure out the estimated fix interpolated along the seg
                pr["estimated_time"] = traj.segment_start
                pr["geometry"] = traj.geometry
                pr["spatialfeature_id"] = list(itertools.repeat(sf.unique_id, pr.shape[0]))
                pr["spatialfeature_name"] = list(itertools.repeat(sf.name, pr.shape[0]))

                proximity_events.append(pr)

        trajectory.groupby("groupby_col").apply(analysis)
        return pd.concat(proximity_events).reset_index(drop=True)
