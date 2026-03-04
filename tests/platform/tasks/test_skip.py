import geopandas as gpd  # type: ignore [import-untyped]
from ecoscope.platform.tasks.results._ecomap import (
    LayerDefinition,
    create_point_layer,
)
from ecoscope.platform.tasks.skip import (
    all_geometry_are_none,
)
from shapely.geometry import Point
from wt_task import task
from wt_task.skip import (
    SKIP_SENTINEL,
)


def test_skipif_all_geometry_are_none() -> None:
    data = {
        "geometry": [
            None,
            None,
        ],
        "some_string_value": [
            "some_string",
            "some_other_string",
        ],
    }
    gdf = gpd.GeoDataFrame(data)

    result = (
        task(create_point_layer)
        .validate()
        .skipif(conditions=[all_geometry_are_none], unpack_depth=0)
        .call(geodataframe=gdf)
    )
    assert result is SKIP_SENTINEL


def test_skipif_all_geometry_are_empty() -> None:
    data = {
        "geometry": [
            Point(),
            Point(),
        ],
        "some_string_value": [
            "some_string",
            "some_other_string",
        ],
    }
    gdf = gpd.GeoDataFrame(data)

    result = (
        task(create_point_layer)
        .validate()
        .skipif(conditions=[all_geometry_are_none], unpack_depth=0)
        .call(geodataframe=gdf)
    )
    assert result is SKIP_SENTINEL


def test_skipif_all_geometry_are_none_or_empty() -> None:
    data = {
        "geometry": [
            None,
            Point(),
        ],
        "some_string_value": [
            "some_string",
            "some_other_string",
        ],
    }
    gdf = gpd.GeoDataFrame(data)

    result = (
        task(create_point_layer)
        .validate()
        .skipif(conditions=[all_geometry_are_none], unpack_depth=0)
        .call(geodataframe=gdf)
    )
    assert result is SKIP_SENTINEL


def test_not_skipif_all_geometry_are_not_none() -> None:
    data = {
        "geometry": [
            None,
            Point(0.0, 0.0),
        ],
        "some_string_value": [
            "some_string",
            "some_other_string",
        ],
    }
    gdf = gpd.GeoDataFrame(data)

    result = (
        task(create_point_layer)
        .validate()
        .skipif(conditions=[all_geometry_are_none], unpack_depth=0)
        .call(geodataframe=gdf)
    )
    assert isinstance(result, LayerDefinition)
