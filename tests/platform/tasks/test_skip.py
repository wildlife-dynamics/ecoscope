import geopandas as gpd  # type: ignore [import-untyped]
import pandas as pd
from shapely.geometry import Point
from wt_task import task
from wt_task.skip import (
    SKIP_SENTINEL,
)

from ecoscope.platform.tasks.results._ecomap import (
    LayerDefinition,
    create_point_layer,
)
from ecoscope.platform.tasks.skip import (
    all_geometry_are_none,
    all_keyed_iterables_are_skips,
    any_dependency_is_empty_string,
    any_dependency_is_none,
    any_dependency_skipped,
    any_is_empty_df,
    never,
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


def test_any_is_empty_df() -> None:
    assert any_is_empty_df(pd.DataFrame()) is True
    assert any_is_empty_df(pd.DataFrame({"a": [1]})) is False


def test_any_dependency_skipped() -> None:
    assert any_dependency_skipped(1, SKIP_SENTINEL, "x") is True
    assert any_dependency_skipped(1, "x") is False


def test_never_is_always_false() -> None:
    assert never() is False
    assert never(1, 2, 3) is False


def test_all_keyed_iterables_are_skips_true() -> None:
    assert all_keyed_iterables_are_skips([("k1", SKIP_SENTINEL), ("k2", SKIP_SENTINEL)]) is True


def test_all_keyed_iterables_are_skips_false() -> None:
    assert all_keyed_iterables_are_skips([("k1", SKIP_SENTINEL), ("k2", "value")]) is False


def test_any_dependency_is_none() -> None:
    assert any_dependency_is_none(1, None, "x") is True
    assert any_dependency_is_none(1, "x") is False


def test_any_dependency_is_empty_string() -> None:
    assert any_dependency_is_empty_string("a", "", "b") is True
    assert any_dependency_is_empty_string("a", "b") is False
