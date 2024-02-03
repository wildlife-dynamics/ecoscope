import logging
import math
import os
from collections import UserDict

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import rasterio.mask
import tqdm.auto as tqdm
import datashader as ds
from rasterio.plot import reshape_as_raster

logger = logging.getLogger(__name__)


class RasterExtent:
    # minx, miny, maxx, maxy
    def __init__(self, x_min=33.0, x_max=37.0, y_min=2.0, y_max=-2.0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def __repr__(self):
        return "RasterExtent(xmin: {0}, xmax: {1}, ymin: {2}, ymax: {3})".format(
            self.x_min, self.x_max, self.y_min, self.y_max
        )

    @classmethod
    def create_from_origin(cls, pixel_size=0.5, x_min=0.0, y_min=0.0, num_rows=100, num_columns=100):
        max_x = x_min + (num_columns * pixel_size)
        max_y = y_min + (num_rows * pixel_size)
        return cls(x_min=x_min, x_max=max_x, y_min=y_min, y_max=max_y)


class RasterProfile(UserDict):
    """
    A class for holding raster properties
    At present this class is only valid for non-rotated rasters with a north-up orientation and square sized pixels
    defined by the E-W pixel size
    """

    def __init__(
        self,
        pixel_size=1000.0,
        pixel_dtype=rio.float64,
        crs="EPSG:8857",  # WGS 84/Equal Earth Greenwich
        nodata_value=0.0,
        band_count=1,
        raster_extent=None,
    ):
        crs = pyproj.CRS.from_user_input(crs)
        raster_extent = raster_extent or RasterExtent()
        cols = int(math.ceil((raster_extent.x_max - raster_extent.x_min) / pixel_size))
        rows = int(math.ceil((raster_extent.y_max - raster_extent.y_min) / pixel_size))
        affine_transform = rio.Affine(pixel_size, 0.0, raster_extent.x_min, 0.0, -pixel_size, raster_extent.y_max)
        super().__init__(
            self,
            crs=crs,
            pixel_size=pixel_size,
            nodata_value=nodata_value,
            columns=cols,
            rows=rows,
            dtype=pixel_dtype,
            band_count=band_count,
            raster_extent=raster_extent,
            transform=affine_transform,
        )

    def _recompute_transform_(self, key):
        """Recomputes the affine transformation matrix when the pixel_size or raster_extent value is updated."""
        _names = ["pixel_size", "raster_extent"]
        if all([key in _names, hasattr(self, _names[0]), hasattr(self, _names[1])]):
            xmin = self.raster_extent.x_min
            ymax = self.raster_extent.y_max
            pixel_size = self.pixel_size

            transform = rio.Affine(pixel_size, 0.0, xmin, 0.0, -pixel_size, ymax)
            cols = int(math.ceil((self.raster_extent.x_max - self.raster_extent.x_min) / self.pixel_size))
            rows = int(math.ceil((self.raster_extent.y_max - self.raster_extent.y_min) / self.pixel_size))

            self.update(transform=transform, columns=cols, rows=rows)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "data":
            object.__setattr__(self, name, value)
        else:
            self[name] = value
            self._recompute_transform_(name)

    def __setitem__(self, key, item):
        self.data[key] = item
        self._recompute_transform_(key)


class RasterPy:
    @classmethod
    def write(
        cls,
        ndarray,
        fp,
        columns,
        rows,
        band_count,
        driver="GTiff",
        dtype=rio.float64,
        crs=None,
        transform=None,
        nodata=None,
        sharing=False,
        indexes=1,
        **kwargs,
    ):
        with rio.open(
            fp=fp,
            mode="w",
            driver=driver,
            height=rows,
            width=columns,
            count=band_count,
            dtype=dtype,
            crs=crs,
            nodata=nodata,
            sharing=sharing,
            transform=transform,
            **kwargs,
        ) as r:
            r.write(ndarray, indexes=indexes)

    @classmethod
    def read(cls, fp, driver=None, **kwargs):
        fp = os.fspath(fp)
        path = rio.parse_path(fp)
        return rio.DatasetReader(path, driver=driver, **kwargs)

    @classmethod
    def read_write(cls, fp, driver=None, **kwargs):
        fp = os.fspath(fp)
        path = rio.parse_path(fp)
        return rio.get_writer_for_path(path, driver=driver)(path, "r+", driver=driver, **kwargs)


def reduce_region(gdf, raster_path_list, reduce_func):
    """
    A function to apply the reduce_func to the values of the pixels within each of the rasters for every
    shape within the input geopandas dataframe 'geometry' column
    :param gdf: geopandas dataframe. The geometry column will be used to mask the areas of the input
    raster to be used
    in the reduction
    :param raster_path_list: a list of raster files on disc to analyse
    :param reduce_func: a single-value function to apply to the values of the input raster
    :return: dataframe with a column of reduce values for each raster and a row for each region
    """

    reduce_func = {
        np.mean: np.nanmean,
        np.sum: np.nansum,
        np.min: np.nanmin,
        np.max: np.nanmax,
    }.get(reduce_func, reduce_func)

    d = {}
    for raster_path in tqdm.tqdm(raster_path_list):
        d[raster_path] = {}
        with rio.open(raster_path) as src:
            for i, shp in gdf.geometry.to_crs(src.crs).iteritems():
                try:
                    d[raster_path][i] = reduce_func(rio.mask.mask(src, [shp], filled=False)[0].compressed())
                except ValueError as e:
                    logger.exception(raster_path, i, e)

    return pd.DataFrame(d)


def raster_to_gdf(raster_path):
    with rio.open(raster_path) as src:
        image = src.read(1)

        dtype = {
            np.float64: np.float32,
            np.int64: np.int32,
            np.uint64: np.int32,
            np.uint32: np.int32,
        }.get(image.dtype.type, image.dtype)

        if dtype is not image.dtype:
            print(f"Error: {image.dtype} is not supported. Converting to {dtype}. Data may be lost.")
            image = image.astype(dtype)

        mask = src.dataset_mask()
        return gpd.GeoDataFrame.from_features(
            [
                {"properties": {"value": value}, "geometry": polygon}
                for polygon, value in rio.features.shapes(image, transform=src.transform, mask=mask)
                if pd.notna(value)
            ],
            crs=src.crs,
        )


def datashade_gdf(gdf, geom_type, width=600, height=600, cmap=["lightblue", "darkblue"], ds_agg=None, **kwargs):
    """
    Creates a raster of the given gdf using Datashader

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame used to create visualization (geometry must be projected to a CRS)
    geom_type : str
        The Datashader canvas() function to use valid values are 'polygon', 'line', 'point'
    width : int
        The canvas width in pixels, determines the resolution of the generated image
    height : int
        The canvas height in pixels, determines the resolution of the generated image
    cmap : list of colors or matplotlib.colors.Colormap
        The colormap to use for the generated image
    ds_agg : datashader.reductions function
        The Datashader reduction to use
    kwargs
        Additional kwargs passed to datashader.transfer_functions.shade
    """

    gdf = gdf.to_crs(epsg=4326)
    bounds = gdf.geometry.total_bounds
    canvas = ds.Canvas(width, height)

    if geom_type == "polygon":
        func = canvas.polygons
    elif geom_type == "line":
        func = canvas.line
    elif geom_type == "point":
        func = canvas.points
    else:
        raise ValueError("geom_type must be 'polygon', 'line' or 'point'")

    agg = func(gdf, geometry="geometry", agg=ds_agg)
    img = ds.tf.shade(agg, cmap, **kwargs)

    return img.to_pil(), bounds


def export_pil_to_cog(image, bounds, filepath, **kwargs):
    """
    Exports a Pillow image to COG

    Parameters
    ----------
    image : PIL.Image
        The image to export
    bounds: tuple
        Tuple containing the EPSG:4326 (minx, miny, maxx, maxy) values bounding the given image
    filepath : string
        Path of file to be created
    kwargs
        Additional args passed through to rasterio.open()
    """

    data = np.array(image)
    data = reshape_as_raster(data)

    width = data.shape[2]
    height = data.shape[1]
    bands = data.shape[0]

    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    RasterPy.write(
        ndarray=data,
        fp=filepath,
        columns=width,
        rows=height,
        band_count=bands,
        driver="COG",
        dtype="uint8",
        crs=rasterio.CRS.from_epsg(4326),
        nodata=0,
        transform=transform,
        indexes=[i + 1 for i in range(bands)],
        **kwargs,
    )
