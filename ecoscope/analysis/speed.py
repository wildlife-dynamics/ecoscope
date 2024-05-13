import typing

import geopandas as gpd
import pandas as pd
import shapely

import ecoscope.analysis.classifier as classifier
import ecoscope.base

default_speed_colors = [
    "#1a9850",
    "#91cf60",
    "#d9ef8b",
    "#fee08b",
    "#fc8d59",
    "#d73027",
]


class SpeedDataFrame(ecoscope.base.EcoDataFrame):
    @classmethod
    def from_trajectory(
        cls,
        trajectory: ecoscope.base.Trajectory,
        classification_method: str = "equal_interval",
        num_classes: int = 6,
        bins: typing.List = None,
        speed_colors: typing.List = None,
    ):
        if not bins:
            bins = classifier.apply_classification(trajectory.speed_kmhr, num_classes, cls_method=classification_method)

        if not speed_colors:
            speed_colors = classifier.default_speed_colors

        speed_colors = speed_colors[: len(bins) - 1]

        speed_df = cls(
            geometry=gpd.GeoSeries(
                trajectory.geometry.values,
                index=pd.Index(
                    pd.cut(x=trajectory.speed_kmhr, bins=bins, labels=speed_colors, include_lowest=True),
                    name="speed_colour",
                ),
            )
            .groupby(level=0)
            .apply(lambda gs: shapely.multilinestrings(gs)),
            crs=trajectory.crs,
        )
        speed_df.reset_index(drop=False, inplace=True)
        speed_df["label"] = _speedmap_labels(bins)
        speed_df.sort_values("speed_colour", inplace=True)

        return speed_df


def _speedmap_labels(bins):
    return [f"{bins[i]:.1f} - {bins[i + 1]:.1f} km/hr" for i in range(len(bins) - 1)]
