import mapclassify
import pandas as pd

default_colors = [
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
        The multiples of the standard deviation to add/subtract from the sample mean to define the bins.
        defaults=[-2, -1, 1, 2]
    """

    classifier = classification_methods.get(cls_method)
    if not classifier:
        return

    map_classifier = classifier(x, multiples) if cls_method == "std_mean" else classifier(x, k)
    edges, _, _ = mapclassify.classifiers._format_intervals(map_classifier, fmt="{:.2f}")
    return [float(i) for i in edges]


def set_label_index(df, bins, column, labels, include_lowest=True):
    # Bin values into discrete intervals
    index = pd.cut(x=df[column], bins=bins, labels=labels, include_lowest=include_lowest)
    df.reset_index(drop=False, inplace=True)
    df.set_index(index, inplace=True)
