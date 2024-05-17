import mapclassify

classification_methods = {
    "equal_interval": mapclassify.EqualInterval,
    "natural_breaks": mapclassify.NaturalBreaks,
    "quantile": mapclassify.Quantiles,
    "std_mean": mapclassify.StdMean,
    "max_breaks": mapclassify.MaximumBreaks,
    "fisher_jenks": mapclassify.FisherJenks,
}


# pass in a series and output the series
def apply_classification(x, labels=None, scheme="natural_breaks", **kwargs):
    """
    Classifies the data in a GeoDataFrame column using specified classification scheme.

    Args:
    y : An array containing the data to classify.
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

    classifier = classifier_class(x, **kwargs)
    if labels is None:
        labels = classifier.bins
    assert len(labels) == len(classifier.bins)
    return [labels[i] for i in classifier.yb]
