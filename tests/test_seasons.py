import geopandas as gpd
import pytest

from ecoscope import plotting
from ecoscope.analysis import seasons

if not pytest.earthengine:
    pytest.skip(
        "Skipping tests because connection to Earth Engine is not available.",
        allow_module_level=True,
    )


@pytest.mark.skipif(not pytest.earthengine, reason="No connection to EarthEngine.")
def test_seasons():
    gdf = gpd.read_file("tests/sample_data/vector/AOI_sites.gpkg").to_crs(4326)

    aoi = gdf.geometry.iat[0]

    # Extract the standardized NDVI ndvi_vals within the AOI
    ndvi_vals = seasons.std_ndvi_vals(
        aoi,
        img_coll="MODIS/061/MCD43A4",
        nir_band="Nadir_Reflectance_Band2",
        red_band="Nadir_Reflectance_Band1",
        start="2010-01-01",
        end="2021-01-01",
    )

    # Calculate the seasonal transition point
    cuts = seasons.val_cuts(ndvi_vals, 2)

    # Determine the seasonal time windows
    windows = seasons.seasonal_windows(ndvi_vals, cuts, season_labels=["dry", "wet"])

    plotting.plot_seasonal_dist(ndvi_vals["NDVI"], cuts)

    assert len(windows) > 0
