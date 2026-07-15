import math

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from shapely.geometry import box

from ecoscope.platform.tasks.analysis import (
    get_density_legend_title,
    get_weighting_column,
    normalize_density_units,
    set_patrol_weighting_spec,
)
from ecoscope.platform.tasks.analysis._density_weighting import labeled_weighting
from ecoscope.platform.tasks.analysis._patrol_density import PATROL_WEIGHTING_SPECS


def test_set_patrol_weighting_spec():
    assert (
        set_patrol_weighting_spec(density_sum_column="timespan_seconds") is PATROL_WEIGHTING_SPECS["timespan_seconds"]
    )
    assert set_patrol_weighting_spec(density_sum_column="dist_meters") is PATROL_WEIGHTING_SPECS["dist_meters"]


def test_get_weighting_column():
    assert get_weighting_column(weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"]) == "timespan_seconds"
    assert get_weighting_column(weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"]) == "dist_meters"


def test_normalize_density_units_time():
    grid = gpd.GeoDataFrame(
        data={"density": [7200.0, 1800.0, np.nan]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"])
    assert math.isclose(result["density"][0], 2.0)
    assert math.isclose(result["density"][1], 0.5)
    assert np.isnan(result["density"][2])


def test_normalize_density_units_distance():
    grid = gpd.GeoDataFrame(
        data={"density": [2500.0, 750.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"])
    assert math.isclose(result["density"][0], 2.5)
    assert math.isclose(result["density"][1], 0.75)


def test_get_density_legend_title():
    assert get_density_legend_title(weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"]) == "Time (h)"
    assert get_density_legend_title(weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"]) == "Distance (km)"


def test_labeled_weighting_replaces_enum_with_labeled_options():
    schema = {"enum": ["timespan_seconds", "dist_meters"]}
    labeled_weighting(PATROL_WEIGHTING_SPECS)(schema)
    assert "enum" not in schema
    assert schema["oneOf"] == [
        {"const": "timespan_seconds", "title": "Time"},
        {"const": "dist_meters", "title": "Distance"},
    ]
