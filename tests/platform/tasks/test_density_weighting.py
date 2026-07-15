import math

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from shapely.geometry import box

from ecoscope.platform.tasks.analysis import (
    get_density_legend_title,
    normalize_density_units,
    set_density_weighting,
)
from ecoscope.platform.tasks.analysis._density_weighting import labeled_weighting
from ecoscope.platform.tasks.analysis._patrol_density import PATROL_WEIGHTING_SPECS


def test_set_density_weighting():
    assert set_density_weighting(density_sum_column="timespan_seconds") == "timespan_seconds"
    assert set_density_weighting(density_sum_column="dist_meters") == "dist_meters"


def test_normalize_density_units_time():
    grid = gpd.GeoDataFrame(
        data={"density": [7200.0, 1800.0, np.nan]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, density_sum_column="timespan_seconds")
    assert math.isclose(result["density"][0], 2.0)
    assert math.isclose(result["density"][1], 0.5)
    assert np.isnan(result["density"][2])


def test_normalize_density_units_distance():
    grid = gpd.GeoDataFrame(
        data={"density": [2500.0, 750.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, density_sum_column="dist_meters")
    assert math.isclose(result["density"][0], 2.5)
    assert math.isclose(result["density"][1], 0.75)


def test_get_density_legend_title():
    assert get_density_legend_title(density_sum_column="timespan_seconds") == "Patrol Effort (h)"
    assert get_density_legend_title(density_sum_column="dist_meters") == "Patrol Effort (km)"
    assert (
        get_density_legend_title(density_sum_column="dist_meters", title_prefix="Ranger Effort") == "Ranger Effort (km)"
    )


def test_labeled_weighting_replaces_enum_with_labeled_options():
    schema = {"enum": ["timespan_seconds", "dist_meters"]}
    labeled_weighting(PATROL_WEIGHTING_SPECS)(schema)
    assert "enum" not in schema
    assert schema["oneOf"] == [
        {"const": "timespan_seconds", "title": "Time"},
        {"const": "dist_meters", "title": "Distance"},
    ]
