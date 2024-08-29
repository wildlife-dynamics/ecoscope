import pandas as pd

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
    dataframe (pd.DatFrame): The frame of data
    column_name (str): The datafraem column to classify.
    labels (str): labels of bins, use bin edges if labels==None.
    scheme (str): Classification scheme to use [equal_interval, natural_breaks, quantile, std_mean, max_breaks,
    fisher_jenks]

    **kwargs: Additional keyword arguments specific to the classification scheme.

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
