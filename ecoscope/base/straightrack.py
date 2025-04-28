import geopandas as gpd  # type: ignore[import-untyped]
from pyproj import Geod


class StraightTrackProperties:
    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf

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
    def nsd(self):
        start_point = self.gdf["geometry"].iloc[0]
        geod = Geod(ellps="WGS84")
        geod_displacement = [geod.inv(start_point.x, start_point.y, geo.x, geo.y)[2] for geo in self.gdf["_geometry"]]
        return [(x**2) / (1000 * 2) for x in geod_displacement]

    @property
    def timespan_seconds(self):
        return (self.gdf["_fixtime"] - self.gdf["fixtime"]).dt.total_seconds()

    @property
    def speed_kmhr(self):
        return (self.dist_meters / self.timespan_seconds) * 3.6
