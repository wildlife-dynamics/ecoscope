import warnings

import astroplan
import astropy.coordinates
import astropy.time
import numpy as np
import pyproj


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
