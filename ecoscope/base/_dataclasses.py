import datetime
import uuid
from dataclasses import dataclass
from typing import Any

import geopandas  # type: ignore[import-untyped]
import shapely
import shapely.geometry


@dataclass
class RelocsCoordinateFilter:
    """Filter parameters for filtering get_fixes based on X/Y coordinate ranges or specific coordinate values"""

    min_x: float = -180.0
    max_x: float = 180.0
    min_y: float = -90.0
    max_y: float = 90.0
    filter_point_coords: list[list[float]] | geopandas.GeoSeries | None = None

    def __post_init__(self):
        if isinstance(self.filter_point_coords, list):
            self.filter_point_coords = geopandas.GeoSeries(
                shapely.geometry.Point(coord) for coord in self.filter_point_coords
            )
        if self.filter_point_coords is None:
            self.filter_point_coords = [[0.0, 0.0]]


@dataclass
class RelocsDateRangeFilter:
    """
    Filter parameters for filtering based on a datetime range
    """

    start: datetime.datetime
    end: datetime.datetime


@dataclass
class RelocsSpeedFilter:
    """
    Filter parameters for filtering based on the speed needed to move from one fix to the next
    """

    max_speed_kmhr: float = float("inf")
    temporal_order: str = "ASC"


@dataclass
class RelocsDistFilter:
    """
    Filter based on the distance between consecutive fixes. Fixes are filtered to the range [min_dist_km, max_dist_km].
    """

    min_dist_km: float = 0.0
    max_dist_km: float = float("inf")
    temporal_order: str = "ASC"


@dataclass
class TrajSegFilter:
    """
    Class filtering a set of trajectory segment segments
    """

    min_length_meters: float = 0.0
    max_length_meters: float = float("inf")
    min_time_secs: float = 0.0
    max_time_secs: float = float("inf")
    min_speed_kmhr: float = 0.0
    max_speed_kmhr: float = float("inf")


@dataclass
class SpatialFeature:
    """
    A spatial geometry with an associated name and unique ID. Becomes a useful construct in several movdata calculations
    """

    name: str = ""
    unique_id: Any = uuid.uuid4()
    geometry: Any = None


@dataclass
class ProximityProfile:
    spatial_features: list[SpatialFeature]
