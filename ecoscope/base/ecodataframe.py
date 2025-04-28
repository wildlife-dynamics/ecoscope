import warnings
from copy import deepcopy

import geopandas as gpd  # type: ignore[import-untyped]


class EcoDataFrame:
    """
    `EcoDataFrame` wraps `geopandas.GeoDataFrame` to provide customizations and allow for simpler extension.
    """

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf

    @classmethod
    def from_file(cls, filename, **kwargs):
        result = gpd.GeoDataFrame.from_file(filename, **kwargs)
        return cls(result)

    @classmethod
    def from_features(cls, features, **kwargs):
        result = gpd.GeoDataFrame.from_features(features, **kwargs)
        return cls(result)

    def reset_filter(self, inplace=False):
        if inplace:
            frame = self
        else:
            frame = deepcopy(self)

        frame.gdf["junk_status"] = False

        if not inplace:
            return frame

    def remove_filtered(self, inplace=False):
        if inplace:
            frame = self
        else:
            frame = deepcopy(self)

        if not frame.gdf["junk_status"].dtype == bool:
            warnings.warn(
                f"junk_status column is of type {frame.gdf['junk_status'].dtype}, expected `bool`. "
                "Attempting to automatically convert."
            )
            frame.gdf["junk_status"] = frame.gdf["junk_status"].astype(bool)

        frame.gdf.query("~junk_status", inplace=True)

        if not inplace:
            return frame
