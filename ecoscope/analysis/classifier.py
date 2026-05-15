from typing import Literal, TypeAlias

ColorValue: TypeAlias = str | float
HexColor: TypeAlias = str

import geopandas as gpd  # type: ignore[import-untyped]
import matplotlib as mpl
import numpy as np
import pandas as pd

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

    if dataframe[input_column_name].nunique(dropna=False) == 1:
        dataframe[output_column_name] = labels[0] if labels else dataframe[input_column_name]
        return dataframe

    classifier_class = classification_methods.get(scheme)

    if not classifier_class:
        raise ValueError(f"Invalid classification scheme. Choose from: {list(classification_methods.keys())}")

    classifier = classifier_class(dataframe[input_column_name].to_numpy(), **kwargs)
    if labels is None:
        labels = classifier.bins

        if label_ranges and pd.api.types.is_numeric_dtype(dataframe[input_column_name]):
            # We could do this using mapclassify.get_legend_classes, but this generates a cleaner label
            def create_range_label(lower, upper):
                lower = f"{lower:.{label_decimals}f}"
                upper = f"{upper:.{label_decimals}f}"
                return lower if lower == upper else f"{lower} - {upper}"

            ranges = []
            if dataframe[input_column_name].min() != labels[0]:
                ranges.append(create_range_label(dataframe[input_column_name].min(), labels[0]))
            ranges.extend([create_range_label(labels[i], labels[i + 1]) for i in range(len(labels) - 1)])
            labels = ranges
        else:
            labels = [round(label, label_decimals) if label_decimals > 0 else round(label) for label in labels]  # type: ignore[arg-type]

    assert len(labels) == len(classifier.bins)
    if label_prefix or label_suffix:
        labels = [f"{label_prefix}{label}{label_suffix}" for label in labels]
    dataframe[output_column_name] = np.asarray(labels, dtype=object)[classifier.yb]
    return dataframe


def apply_color_map(
    dataframe: pd.DataFrame,
    input_column_name: str,
    cmap: str | list[HexColor] | dict[ColorValue, HexColor],
    output_column_name: str | None = None,
) -> pd.DataFrame:
    """
    Creates a new column on the provided dataframe with the given cmap applied over the specified input column

    Args:
    dataframe (pd.DatFrame): The data.
    input_column_name (str): The dataframe column who's values will be inform the cmap values.
    cmap (str, list, dict): Either a named mpl.colormap, a list of string hex values, or a dict mapping
        values to hex color strings. When a dict is provided, each key is a data value and each value is
        a hex color string (e.g. {"stop": "#FF0000", "go": "#00FF00"}). Data values not present in the
        dict are given set as fully transparent.
    output_column_name(str): The dataframe column that will contain the classification.
        Defaults to "<input_column_name>_colormap"

    Returns:
    The input dataframe with a color map appended.
    """
    assert input_column_name in dataframe.columns, "input column must exist on dataframe"

    s = dataframe[input_column_name]
    NAN_COLOR = (0, 0, 0, 0)

    unique_non_na = s.dropna().unique()
    k = len(unique_non_na)

    if k == 0:
        # column all-NaN/None → fill transparent
        cmap_series = pd.Series(dtype="object")  # empty, we’ll only use NAN_COLOR
    elif isinstance(cmap, list):
        rgba = [hex_to_rgba(x) for x in cmap]
        normalized = [rgba[i % len(rgba)] for i in range(k)]
        cmap_series = pd.Series(normalized, index=unique_non_na)
    elif isinstance(cmap, str):
        mpl_cmap = mpl.colormaps[cmap]
        # numeric vs categorical handling (on non-null values only)
        if pd.api.types.is_numeric_dtype(s.dtype):
            arr = np.asarray(unique_non_na, dtype=float)
            val_min = arr.min()
            val_max = arr.max()
            value_range = 1.0 if val_min == val_max else (val_max - val_min)
            cmap_colors = mpl_cmap((arr - val_min) / value_range)
        else:
            # categorical/string: cycle through the colormap
            if k < mpl_cmap.N:
                mpl_cmap = mpl_cmap.resampled(max(k, 1))
            cmap_colors = mpl_cmap(np.arange(k) % mpl_cmap.N)

        scaled = np.rint(np.asarray(cmap_colors) * 255).astype(int).tolist()
        color_list = [tuple(row) for row in scaled]
        cmap_series = pd.Series(color_list, index=unique_non_na)
    elif isinstance(cmap, dict):
        cmap_series = pd.Series({k: hex_to_rgba(v) for k, v in cmap.items()})
    else:
        raise TypeError(
            "cmap must be a matplotlib colormap name (str), "
            "a list of hex colors, or a dict mapping values to hex colors"
        )

    if not output_column_name:
        output_column_name = f"{input_column_name}_colormap"

    mapped = s.map(cmap_series)
    nan_mask = mapped.isna().to_numpy()
    if nan_mask.any():
        mapped_arr = mapped.to_numpy()
        dataframe[output_column_name] = [NAN_COLOR if nan_mask[i] else mapped_arr[i] for i in range(len(mapped_arr))]
    else:
        dataframe[output_column_name] = mapped
    return dataframe


def classify_percentile(
    df: pd.DataFrame | gpd.GeoDataFrame,
    percentile_levels: list[int],
    input_column_name: str,
    output_column_name: str = "percentile",
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Creates a new column on the provided dataframe with the percentile bin of the input_column
    Uses much the same methodology as `get_percentile_area` but applies
    generally to a numeric dataframe column instead of a raster grid

    Args:
    df (pd.DataFrame | gpd.GeoDatFrame): The data.
    percentile_levels (list[int]): list of k-th percentile scores.
    input_column_name (str): The column to apply classification to.
    output_column_name (str): The dataframe column that will contain the classification.
        Defaults to "percentile"

    Returns:
    The input dataframe with percentile classification appended.
    """
    assert pd.api.types.is_numeric_dtype(df[input_column_name]), "input column must contain numeric values"

    if not percentile_levels:
        return df

    input_values = df[input_column_name].to_numpy()
    input_values = np.sort(input_values[~np.isnan(input_values)])
    csum = np.cumsum(input_values)

    percentile_values = np.array(
        [input_values[np.argmin(np.abs(csum[-1] * (1 - p / 100) - csum))] for p in percentile_levels]
    )

    levels = np.asarray(percentile_levels)
    values = df[input_column_name].to_numpy()
    mask = values[:, None] >= percentile_values[None, :]
    df[output_column_name] = np.where(mask.any(axis=1), levels[mask.argmax(axis=1)], np.nan)

    return df
