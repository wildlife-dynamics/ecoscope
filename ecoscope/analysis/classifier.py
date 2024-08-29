import pandas as pd
import matplotlib as mpl

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
def apply_classification(dataframe, column_name, labels=None, scheme="natural_breaks", **kwargs):
    """
    Classifies the data in a GeoDataFrame column using specified classification scheme.

    Args:
    dataframe (pd.DatFrame): The data.
    column_name (str): The dataframe column to classify.
    labels (str): labels of bins, use bin edges if labels==None.
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
    result: an array of corresponding labels of the input data.
    """

    classifier_class = classification_methods.get(scheme)

    if not classifier_class:
        raise ValueError(f"Invalid classification scheme. Choose from: {list(classification_methods.keys())}")

    classifier = classifier_class(dataframe[column_name].to_numpy(), **kwargs)
    if labels is None:
        labels = classifier.bins
    assert len(labels) == len(classifier.bins)
    classified = [labels[i] for i in classifier.yb]
    return pd.Series(classified, index=dataframe.index)


def create_color_dict(series, cmap, labels=None):

    if isinstance(cmap, list):
        assert len(cmap) == series.nunique()
        cmap = pd.Series(cmap, index=series.unique())
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
        cmap = cmap.resampled(series.nunique())
        cmap = pd.Series([color for color in cmap.colors], index=series.unique())

    vals = dict([(classification, cmap[classification]) for classification in series.values])

    return vals
