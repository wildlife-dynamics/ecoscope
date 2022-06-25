import logging

import ee
import numpy as np
import pandas
import pandas as pd
import shapely
import sklearn.mixture
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def _min_max_scaler(x):
    x = np.array(x, dtype=np.float64)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_std = (x - x_min) / (x_max - x_min)  # in the range 0..1
    return x_std


def std_ndvi_vals(aoi=None, start=None, end=None):
    coll = ee.ImageCollection("MODIS/MCD43A4_006_NDVI").select("NDVI").filterDate(start, end)

    ndvi_vals = (
        coll.toBands()
        .reduceRegion("mean", ee.Feature(shapely.geometry.mapping(aoi)).geometry(), bestEffort=True)
        .values()
        .getInfo()
    )
    ndvi_vals_std = _min_max_scaler(ndvi_vals)
    return pd.DataFrame(
        {
            "img_date": pd.to_datetime(coll.aggregate_array("system:time_start").getInfo(), unit="ms", utc=True),
            "NDVI": ndvi_vals_std,
        }
    ).dropna(axis=0)


def val_cuts(vals, num_seasons=2):
    distr = sklearn.mixture.GaussianMixture(n_components=num_seasons, max_iter=500)
    vals = vals["NDVI"].to_numpy().reshape(-1, 1)
    distr.fit(vals)
    mu_vars = np.array(
        sorted(
            zip(distr.means_.flatten(), distr.covariances_.flatten()),
            key=lambda x: x[0],
        )
    )
    cuts = []
    x = np.linspace(0, 1.0, 1000)
    for i in range(1, len(mu_vars)):
        cuts.append(
            np.max(
                x[
                    tuple(
                        [
                            norm.sf(
                                x,
                                loc=mu_vars[tuple([i - 1, 0])],
                                scale=np.sqrt(mu_vars[tuple([i - 1, 1])]),
                            )
                            > norm.cdf(
                                x,
                                loc=mu_vars[tuple([i, 0])],
                                scale=np.sqrt(mu_vars[tuple([i, 1])]),
                            )
                        ]
                    )
                ]
            )
        )
    cuts.append(float("inf"))
    cuts.append(float("-inf"))
    return sorted(cuts)


def seasonal_windows(ndvi_vals, cuts, season_labels):

    enc = LabelEncoder()
    ndvi_vals["season"] = pandas.cut(ndvi_vals["NDVI"], bins=cuts, labels=season_labels)
    ndvi_vals["season_code"] = enc.fit_transform(ndvi_vals["season"])
    ndvi_vals["unique_season"] = (ndvi_vals["season_code"].diff(1) != 0).astype("int").cumsum()
    ndvi_vals["end"] = ndvi_vals["img_date"].shift(-1)

    # Note: There is a slight shift that needs to be done with the date-ranges returned by GEE. What happens is that
    # the image daterange use 12:00 a.m on the given day rather than midnight so it introduces day long gaps in the
    # time windows. Therefore need to add 1 day to each of the end dates to make them align exactly with the next
    # successive start date
    grpd = ndvi_vals.groupby("unique_season")
    return pandas.DataFrame(
        {
            "start": pandas.Series([grp["img_date"].iloc[0] for _, grp in grpd]),
            "end": pandas.Series([grp["img_date"].iloc[-1] for _, grp in grpd]).apply(
                lambda x: x + pd.Timedelta(days=1)
            ),
            "season": pandas.Series(grp["season"].iloc[0] for name, grp in grpd),
            "unique_season": pandas.Series([name for name, _ in grpd]),
        }
    ).set_index("unique_season")
