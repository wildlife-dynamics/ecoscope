import logging
from datetime import datetime

import ee
import numpy as np
import pandas as pd
import geopandas as gpd  # type: ignore[import-untyped]
import shapely
from shapely.geometry.base import BaseGeometry

try:
    import sklearn.mixture  # type: ignore[import-untyped]
    from scipy.stats import norm  # type: ignore[import-untyped]
    from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["analysis"]'
    )

logger = logging.getLogger(__name__)


def _min_max_scaler(x):
    x = np.array(x, dtype=np.float64)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_std = (x - x_min) / (x_max - x_min)  # in the range 0..1
    return x_std


def std_ndvi_vals(
    aoi: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
    img_coll: str,
    start: str | datetime,
    nir_band: str | None = None,
    red_band: str | None = None,
    end: str | datetime | None = None,
    img_scale: int = 1,
) -> pd.DataFrame:
    coll = (
        ee.ImageCollection(img_coll)
        .select([nir_band, red_band])
        .filterDate(start, end)
        .map(lambda x: x.multiply(ee.Image(img_scale)).set("system:time_start", x.get("system:time_start")))
    )

    if aoi:
        geo = ee.Feature(shapely.geometry.mapping(aoi)).geometry()
    else:
        geo = None

    img_dates = pd.to_datetime(coll.aggregate_array("system:time_start").getInfo(), unit="ms", utc=True)

    coll = coll.map(lambda x: x.normalizedDifference([nir_band, red_band]))
    ndvi_vals = coll.toBands().reduceRegion("mean", geo, bestEffort=True).values().getInfo()

    df = pd.DataFrame(
        {
            "img_date": img_dates,
            "NDVI": ndvi_vals,
        }
    ).dropna(axis=0)

    df["NDVI"] = _min_max_scaler(df["NDVI"])
    return df


def val_cuts(
    vals: pd.DataFrame,
    num_seasons: int = 2,
) -> list[float]:
    distr = sklearn.mixture.GaussianMixture(n_components=num_seasons, max_iter=500)
    ndvi_vals = vals["NDVI"].to_numpy().reshape(-1, 1)
    distr.fit(ndvi_vals)
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


def seasonal_windows(
    ndvi_vals: pd.DataFrame,
    cuts: list[float],
    season_labels: list[str],
) -> pd.DataFrame:
    enc = LabelEncoder()
    ndvi_vals["season"] = pd.cut(ndvi_vals["NDVI"], bins=cuts, labels=season_labels)
    ndvi_vals["season_code"] = enc.fit_transform(ndvi_vals["season"])
    ndvi_vals["unique_season"] = (ndvi_vals["season_code"].diff(1) != 0).astype("int").cumsum()
    ndvi_vals["end"] = ndvi_vals["img_date"].shift(-1)

    # Note: There is a slight shift that needs to be done with the date-ranges returned by GEE. What happens is that
    # the image daterange use 12:00 a.m on the given day rather than midnight so it introduces day long gaps in the
    # time windows. Therefore need to add 1 day to each of the end dates to make them align exactly with the next
    # successive start date
    grpd = ndvi_vals.groupby("unique_season")
    return pd.DataFrame(
        {
            "start": pd.Series([grp["img_date"].iloc[0] for _, grp in grpd]),
            "end": pd.Series([grp["img_date"].iloc[-1] for _, grp in grpd]).apply(lambda x: x + pd.Timedelta(days=1)),
            "season": pd.Series([grp["season"].iloc[0] for name, grp in grpd]),
            "unique_season": pd.Series([name for name, _ in grpd]),
        }
    ).set_index("unique_season")


def add_seasonal_index(
    df: pd.DataFrame,
    index_name: str,
    start_date: datetime,
    end_date: datetime,
    time_col: str,
    ndvi_vals: gpd.GeoDataFrame,
    seasons: int = 2,
    season_labels: list[str] | None = None,
) -> pd.DataFrame:
    if season_labels is None:
        season_labels = ["dry", "wet"]

    if len(season_labels) != seasons:
        raise Exception(
            f"Parameter value 'seasons' ({seasons}) must match the number of 'season_labels' elements ({season_labels})"
        )

    # calculate the seasonal transition point
    cuts = val_cuts(ndvi_vals, seasons)

    # determine the seasonal time windows
    windows = seasonal_windows(ndvi_vals, cuts, season_labels)

    # Categorize the fixtime values according to the season
    bins: pd.IntervalIndex = pd.IntervalIndex(data=windows.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1))  # type: ignore[arg-type]
    labels = windows.season
    df[index_name] = pd.cut(df[time_col], bins=bins).map(dict(zip(bins, labels)))

    # set the index
    return df.set_index(index_name, append=True)
