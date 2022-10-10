import typing

import geopandas as gpd
import mapclassify
import pandas as pd
import pygeos

import ecoscope.base


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
            bins = apply_classification(trajectory.speed_kmhr, num_classes, cls_method=classification_method)

        if not speed_colors:
            speed_colors = default_speed_colors

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
            .apply(lambda gs: pygeos.multilinestrings(gs.values.data)),
            crs=trajectory.crs,
        )
        speed_df.reset_index(drop=False, inplace=True)
        speed_df["label"] = _speedmap_labels(bins)
        speed_df.sort_values("speed_colour", inplace=True)

        return speed_df


def _speedmap_labels(bins):
    return [f"{bins[i]:.1f} - {bins[i + 1]:.1f} km/hr" for i in range(len(bins) - 1)]


def apply_classification(x, k, cls_method="natural_breaks", multiples=[-2, -1, 1, 2]):
    """
    Function to select which classifier to apply to the speed distributed data.

    Parameters
    ----------
    x : np.ndarray
        The input array to be classified. Must be 1-dimensional
    k : int
        Number of classes required.
    cls_method : str
        Classification method
    multiples : Listlike
        The multiples of the standard deviation to add/subtract from the sample mean to define the bins. defaults=
    """

    classifier = classification_methods.get(cls_method)
    if not classifier:
        return

    map_classifier = classifier(x, multiples) if cls_method == "std_mean" else classifier(x, k)
    edges, _, _ = mapclassify.classifiers._format_intervals(map_classifier, fmt="{:.2f}")
    return [float(i) for i in edges]


default_speed_colors = [
    "#1a9850",
    "#91cf60",
    "#d9ef8b",
    "#fee08b",
    "#fc8d59",
    "#d73027",
]

classification_methods = {
    "equal_interval": mapclassify.EqualInterval,
    "natural_breaks": mapclassify.NaturalBreaks,
    "quantile": mapclassify.Quantiles,
    "std_mean": mapclassify.StdMean,
    "max_breaks": mapclassify.MaximumBreaks,
    "fisher_jenks": mapclassify.FisherJenks,
}
