import warnings
from datetime import datetime, timedelta
from shapely import Geometry
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import pytz

try:
    import astroplan
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["analysis"]'
    )


def to_EarthLocation(geometry: gpd.GeoSeries) -> EarthLocation:
    """
    Location on Earth, initialized from geocentric coordinates.

    Parameters
    ----------
    geometry: Geometry
        GeoDataFrame's geometry column

    Returns
    -------
    astropy.coordinates.EarthLocation.
    """
    geometry = geometry.to_crs(4326)
    trans = pyproj.Transformer.from_proj(
        proj_from="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        proj_to="+proj=geocent +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
    )
    return EarthLocation.from_geocentric(
        *trans.transform(xx=geometry.x, yy=geometry.y, zz=np.zeros(geometry.shape[0])), unit="m"
    )


def is_night(geometry: gpd.GeoSeries, time: pd.Series) -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.", UserWarning)
        return astroplan.Observer(to_EarthLocation(geometry.centroid)).is_night(time)


def sun_time(date: datetime, geometry: Geometry) -> pd.Series:
    midnight = Time(
        datetime(date.year, date.month, date.day) + timedelta(seconds=1), scale="utc"
    )  # add 1 second shift to avoid leap_second_strict warning
    observer = astroplan.Observer(location=EarthLocation(lon=geometry.centroid.x, lat=geometry.centroid.y))
    sunrise = observer.sun_rise_time(midnight, which="next", n_grid_points=150).to_datetime(timezone=pytz.UTC)
    sunset = observer.sun_set_time(midnight, which="next", n_grid_points=150).to_datetime(timezone=pytz.UTC)
    return pd.Series({"sunrise": sunrise, "sunset": sunset})


def calculate_day_night_distance(
    date: datetime, segment_start: datetime, segment_end: datetime, dist_meters: int, daily_summary: pd.DataFrame
) -> None:
    sunrise = daily_summary.loc[date, "sunrise"]
    sunset = daily_summary.loc[date, "sunset"]

    if segment_start < sunset and segment_end > sunset:  # start in day and end in night
        day_percent = (sunset - segment_start) / (segment_end - segment_start)
    elif segment_start < sunrise and segment_end > sunrise:  # start in night and end in day
        day_percent = (segment_end - sunrise) / (segment_end - segment_start)
    elif sunrise < sunset:
        if segment_end < sunrise or segment_start > sunset:  # all night
            day_percent = 0
        elif segment_start >= sunrise and segment_end <= sunset:  # all day
            day_percent = 1
    else:  # sunrise >= sunset
        if segment_end < sunset or segment_start > sunrise:  # all day
            day_percent = 1
        elif segment_start >= sunset and segment_end <= sunrise:  # all night
            day_percent = 0

    daily_summary.loc[date, "day_distance"] += day_percent * dist_meters
    daily_summary.loc[date, "night_distance"] += (1 - day_percent) * dist_meters


def get_nightday_ratio(gdf: gpd.GeoDataFrame) -> float:
    gdf["date"] = pd.to_datetime(gdf["segment_start"]).dt.date

    daily_summary = gdf.groupby("date").first()["geometry"].reset_index()
    daily_summary[["sunrise", "sunset"]] = daily_summary.apply(lambda x: sun_time(x.date, x.geometry), axis=1)
    daily_summary["day_distance"] = 0.0
    daily_summary["night_distance"] = 0.0
    daily_summary = daily_summary.set_index("date")

    gdf.apply(
        lambda x: calculate_day_night_distance(x.date, x.segment_start, x.segment_end, x.dist_meters, daily_summary),
        axis=1,
    )

    daily_summary["night_day_ratio"] = daily_summary["night_distance"] / daily_summary["day_distance"]
    mean_night_day_ratio = daily_summary["night_day_ratio"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    return mean_night_day_ratio
