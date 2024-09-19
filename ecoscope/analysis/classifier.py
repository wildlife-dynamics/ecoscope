import pandas as pd
import matplotlib as mpl
from ecoscope.base.utils import hex_to_rgba

# from ecoscope.base._dataclasses import ColorStyleLookup

try:
    import mapclassify
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
    dataframe, input_column_name, output_column_name=None, labels=None, scheme="natural_breaks", **kwargs
):
    """
    Classifies the data in a GeoDataFrame column using specified classification scheme.

    Args:
    dataframe (pd.DatFrame): The data.
    input_column_name (str): The dataframe column to classify.
    output_column_name (str): The dataframe column that will contain the classification.
        Defaults to "<input_column_name>_classified"
    labels (list[str]): labels of bins, use bin edges if labels==None.
    scheme (str): Classification scheme to use [equal_interval, natural_breaks, quantile, std_mean, max_breaks,
    fisher_jenks]

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
    assert len(labels) == len(classifier.bins)
    dataframe[output_column_name] = [labels[i] for i in classifier.yb]
    return dataframe


def apply_color_map(dataframe, input_column_name, cmap, output_column_name=None):
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

    if isinstance(cmap, list):
        nunique = dataframe[input_column_name].nunique()
        assert len(cmap) >= nunique, f"cmap list must contain at least as many values as unique in {input_column_name}"
        cmap = [hex_to_rgba(x) for x in cmap]
        cmap = pd.Series(cmap[:nunique], index=dataframe[input_column_name].unique())
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
        cmap = cmap.resampled(dataframe[input_column_name].nunique())

        if pd.api.types.is_numeric_dtype(dataframe[input_column_name].dtype):
            cmap_colors = []
            val_min = dataframe[input_column_name].min()
            val_max = dataframe[input_column_name].max()
            for val in dataframe[input_column_name].unique():
                cmap_colors.append(cmap((val - val_min) / (val_max - val_min)))
        else:
            cmap_colors = cmap(range(len(dataframe[input_column_name].unique())))

        color_list = []
        for color in cmap_colors:
            color_list.append(tuple([round(val * 255) for val in color]))

        # convert to hex first to put values in range(0,255), then to an RGBA tuple
        cmap = pd.Series(
            color_list,
            index=dataframe[input_column_name].unique(),
        )

    if not output_column_name:
        output_column_name = f"{input_column_name}_colormap"

    dataframe[output_column_name] = [cmap[classification] for classification in dataframe[input_column_name]]
    return dataframe
