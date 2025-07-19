from typing import Literal
import pandas as pd
import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import matplotlib as mpl
from ecoscope.base.utils import hex_to_rgba


try:
    import mapclassify  # type: ignore[import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["analysis"]'
    )


classification_methods = {
    "equal_interval": mapclassify.EqualInterval,
    "natural_breaks": mapclassify.NaturalBreaks,
    "quantile": mapclassify.Quantiles,
    "std_mean": mapclassify.StdMean,
    "max_breaks": mapclassify.MaximumBreaks,
    "fisher_jenks": mapclassify.FisherJenks,
}


# pass in a dataframe and output a series
def apply_classification(
    dataframe: pd.DataFrame,
    input_column_name: str,
    output_column_name: str | None = None,
    labels: list[str] | None = None,
    scheme: Literal[
        "equal_interval", "natural_breaks", "quantile", "std_mean", "max_breaks", "fisher_jenks"
    ] = "natural_breaks",
    label_prefix: str = "",
    label_suffix: str = "",
    label_ranges: bool = False,
    label_decimals: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Classifies the data in a DataFrame column using specified classification scheme.

    Args:
    dataframe (pd.DatFrame): The data.
    input_column_name (str): The dataframe column to classify.
    output_column_names (str): The dataframe column that will contain the classification.
        Defaults to "<input_column_name>_classified"
    labels (list[str]): labels of bins, use bin edges if labels==None.
    scheme (str): Classification scheme to use [equal_interval, natural_breaks, quantile, std_mean, max_breaks,
    fisher_jenks]
    label_prefix (str): Prepends provided string to each label
    label_suffix (str): Appends provided string to each label
    label_ranges (bool): Applicable only when 'labels' is not set
                         If True, generated labels will be the range between bin edges,
                         rather than the bin edges themselves.
    label_decimals (int): Applicable only when 'labels' is not set
                          Specifies the number of decimal places in the label


    **kwargs:
        Additional keyword arguments specific to the classification scheme, passed to mapclassify.
        See below

    Applicable to equal_interval, natural_breaks, quantile, max_breaks & fisher_jenks:
    k (int): The number of classes required

    Applicable only to natural_breaks:
    initial (int): The number of initial solutions generated with different centroids.
        The best of initial results are returned.

    Applicable only to max_breaks:
    mindiff (float): The minimum difference between class breaks.

    Applicable only to std_mean:
    multiples (numpy.array): The multiples of the standard deviation to add/subtract
        from the sample mean to define the bins.
    anchor (bool): Anchor upper bound of one class to the sample mean.

    For more information, see https://pysal.org/mapclassify/api.html

    Returns:
    The input dataframe with a classification column appended.
    """
    assert input_column_name in dataframe.columns, "input column must exist on dataframe"
    if not output_column_name:
        output_column_name = f"{input_column_name}_classified"

    classifier_class = classification_methods.get(scheme)

    if not classifier_class:
        raise ValueError(f"Invalid classification scheme. Choose from: {list(classification_methods.keys())}")

    classifier = classifier_class(dataframe[input_column_name].to_numpy(), **kwargs)
    if labels is None:
        labels = classifier.bins

        if label_ranges:
            # We could do this using mapclassify.get_legend_classes, but this generates a cleaner label
            ranges = [f"{dataframe[input_column_name].min():.{label_decimals}f} - {labels[0]:.{label_decimals}f}"]
            ranges.extend(
                [f"{labels[i]:.{label_decimals}f} - {labels[i + 1]:.{label_decimals}f}" for i in range(len(labels) - 1)]
            )
            labels = ranges
        else:
            labels = [round(label, label_decimals) for label in labels]  # type: ignore[arg-type]

    assert len(labels) == len(classifier.bins)
    if label_prefix or label_suffix:
        labels = [f"{label_prefix}{label}{label_suffix}" for label in labels]
    dataframe[output_column_name] = [labels[i] for i in classifier.yb]
    return dataframe


def apply_color_map(
    dataframe: pd.DataFrame, input_column_name: str, cmap: str | list[str], output_column_name: str | None = None
) -> pd.DataFrame:
    """
    Creates a new column on the provided dataframe with the given cmap applied over the specified input column

    Args:
    dataframe (pd.DatFrame): The data.
    input_column_name (str): The dataframe column who's values will be inform the cmap values.
    cmap (str, list): Either a named mpl.colormap or a list of string hex values.
    output_column_name(str): The dataframe column that will contain the classification.
        Defaults to "<input_column_name>_colormap"

    Returns:
    The input dataframe with a color map appended.
    """
    assert input_column_name in dataframe.columns, "input column must exist on dataframe"

    nunique = dataframe[input_column_name].nunique()
    unique = dataframe[input_column_name].unique()
    if isinstance(cmap, list):
        rgba = [hex_to_rgba(x) for x in cmap]
        normalized_cmap = [rgba[i % len(rgba)] for i in range(nunique)]
        cmap_series = pd.Series(normalized_cmap, index=unique)
    if isinstance(cmap, str):
        mpl_cmap = mpl.colormaps[cmap]
        if nunique < mpl_cmap.N:
            mpl_cmap = mpl_cmap.resampled(nunique)

        if pd.api.types.is_numeric_dtype(dataframe[input_column_name].dtype):
            val_min = dataframe[input_column_name].min()
            val_max = dataframe[input_column_name].max()
            value_range = 1 if val_min == val_max else val_max - val_min
            cmap_colors = [mpl_cmap((val - val_min) / value_range) for val in unique]
        else:
            cmap_colors = [mpl_cmap(i % mpl_cmap.N) for i in range(nunique)]

        color_list = []
        for color in cmap_colors:
            color_list.append(tuple([round(val * 255) for val in color]))

        cmap_series = pd.Series(
            color_list,
            index=unique,
        )

    if not output_column_name:
        output_column_name = f"{input_column_name}_colormap"

    dataframe[output_column_name] = [cmap_series[classification] for classification in dataframe[input_column_name]]
    return dataframe


def classify_percentile(
    gdf: gpd.GeoDataFrame,
    percentile_levels: list[int],
    input_column_name: str,
    output_column_name: str = "percentile",
) -> gpd.GeoDataFrame:
    """
    Creates a new column on the provided dataframe with the percentile bin of the input_column
    Uses much the same methodology as `get_percentile_area` but applies
    generally to a numeric dataframe column instead of a raster grid

    Args:
    gdf (gpd.GeoDatFrame): The data.
    percentile_levels (list[int]): list of k-th percentile scores.
    input_column_name (str): The column to apply classification to.
    output_column_name (str): The dataframe column that will contain the classification.
        Defaults to "percentile"

    Returns:
    The input dataframe with percentile classification appended.
    """
    assert pd.api.types.is_numeric_dtype(gdf[input_column_name]), "input column must contain numeric values"

    input_values = gdf[input_column_name].to_numpy()
    input_values = np.sort(input_values[~np.isnan(input_values)])
    csum = np.cumsum(input_values)

    percentile_values = []
    for percentile in percentile_levels:
        percentile_values.append(input_values[np.argmin(np.abs(csum[-1] * (1 - percentile / 100) - csum))])

    def find_percentile(value):
        for i in range(len(percentile_levels)):
            if value >= percentile_values[i]:
                return percentile_levels[i]
        return np.nan

    for i in range(len(percentile_levels)):
        gdf[output_column_name] = gdf[input_column_name].apply(find_percentile)

    return gdf
