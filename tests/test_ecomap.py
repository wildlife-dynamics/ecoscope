import geopandas as gpd
import pandas as pd

import ecoscope


def test_ecomap_base():
    ecoscope.mapping.EcoMap()


def test_add_local_geotiff():
    m = ecoscope.mapping.EcoMap()
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap=None)
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap="jet")


def test_add_gdf():
    m = ecoscope.mapping.EcoMap()
    df = pd.read_feather("tests/sample_data/vector/movbank_data.feather")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.pop("location-long"), df.pop("location-lat")),
        crs=4326,
    )
    m.add_gdf(gdf)
    m.zoom_to_bounds(gdf["geometry"].total_bounds)
    m.zoom_to_gdf(gdf)
    m.add_legend()


def test_add_title():
    m = ecoscope.mapping.EcoMap()
    m.add_title("Map test")


def test_output():
    m = ecoscope.mapping.EcoMap()
    m.to_html("ecomap_html_test.html")


def test_add_arrow():
    m = ecoscope.mapping.EcoMap()
    m.add_north_arrow()
