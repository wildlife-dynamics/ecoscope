import warnings

import astroplan
import astropy.coordinates
import astropy.time
import numpy as np
import pyproj
import pandas as pd


def to_EarthLocation(geometry):
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
    return astropy.coordinates.EarthLocation.from_geocentric(
        *trans.transform(xx=geometry.x, yy=geometry.y, zz=np.zeros(geometry.shape[0])), unit="m"
    )


def is_night(geometry, time):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.", UserWarning)
        return astroplan.Observer(to_EarthLocation(geometry.centroid)).is_night(time)


def get_daynight_ratio(traj, n_grid_points=150) -> pd.Series:
    """
    Parameters
    ----------
    n_grid_points : int (optional)
        The number of grid points on which to search for the horizon
        crossings of the target over a 24-hour period, default is 150 which
        yields rise time precisions better than one minute.
        https://github.com/astropy/astroplan/pull/424
    Returns
    -------
    pd.Series:
        Daynight ratio for each unique individual subject in the grouby_col column.
    """

    locations = to_EarthLocation(traj.geometry.to_crs(crs=traj.estimate_utm_crs()).centroid)

    observer = astroplan.Observer(location=locations)
    is_night_start = observer.is_night(traj.segment_start)
    is_night_end = observer.is_night(traj.segment_end)

    # Night -> Night
    night_distance = traj.dist_meters.loc[is_night_start & is_night_end].sum()

    # Day -> Day
    day_distance = traj.dist_meters.loc[~is_night_start & ~is_night_end].sum()

    # Night -> Day
    night_day_mask = is_night_start & ~is_night_end
    night_day_df = traj.loc[night_day_mask, ["segment_start", "dist_meters", "timespan_seconds"]]
    i = (
        pd.to_datetime(
            astroplan.Observer(location=locations[night_day_mask])
            .sun_rise_time(
                astropy.time.Time(night_day_df.segment_start),
                n_grid_points=n_grid_points,
                which="next",
            )
            .datetime,
            utc=True,
        )
        - night_day_df.segment_start
    ).dt.total_seconds() / night_day_df.timespan_seconds

    night_distance += (night_day_df.dist_meters * i).sum()
    day_distance += ((1 - i) * night_day_df.dist_meters).sum()

    # Day -> Night
    day_night_mask = ~is_night_start & is_night_end
    day_night_df = traj.loc[day_night_mask, ["segment_start", "dist_meters", "timespan_seconds"]]
    i = (
        pd.to_datetime(
            astroplan.Observer(location=locations[day_night_mask])
            .sun_set_time(
                astropy.time.Time(day_night_df.segment_start),
                n_grid_points=n_grid_points,
                which="next",
            )
            .datetime,
            utc=True,
        )
        - day_night_df.segment_start
    ).dt.total_seconds() / day_night_df.timespan_seconds
    day_distance += (day_night_df.dist_meters * i).sum()
    night_distance += ((1 - i) * day_night_df.dist_meters).sum()

    return day_distance / night_distance
