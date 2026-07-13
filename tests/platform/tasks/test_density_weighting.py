import math

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from shapely.geometry import box

from ecoscope.platform.tasks.analysis import (
    get_density_legend_title,
    normalize_density_units,
    set_density_weighting,
)


def test_set_density_weighting():
    assert set_density_weighting(sum_column="timespan_seconds") == "timespan_seconds"
    assert set_density_weighting(sum_column="dist_meters") == "dist_meters"


def test_normalize_density_units_time():
    grid = gpd.GeoDataFrame(
        data={"density": [7200.0, 1800.0, np.nan]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, sum_column="timespan_seconds")
    assert math.isclose(result["density"][0], 2.0)
    assert math.isclose(result["density"][1], 0.5)
    assert np.isnan(result["density"][2])


def test_normalize_density_units_distance():
    grid = gpd.GeoDataFrame(
        data={"density": [2500.0, 750.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, sum_column="dist_meters")
    assert math.isclose(result["density"][0], 2.5)
    assert math.isclose(result["density"][1], 0.75)


def test_get_density_legend_title():
    assert get_density_legend_title(sum_column="timespan_seconds") == "Patrol Effort (hours)"
    assert get_density_legend_title(sum_column="dist_meters") == "Patrol Effort (km)"
    assert get_density_legend_title(sum_column="dist_meters", title_prefix="Ranger Effort") == "Ranger Effort (km)"
