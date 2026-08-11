import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from wt_task.skip import SKIP_SENTINEL

from ecoscope.platform.tasks.transformation import concat_dataframes
from ecoscope.platform.tasks.transformation._concat import _drop_skip_sentinels


def test_concat_basic() -> None:
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [3], "b": ["z"]})

    result = concat_dataframes([df1, df2])

    assert len(result) == 3
    assert list(result["a"]) == [1, 2, 3]
    assert list(result.index) == [0, 1, 2]


def test_concat_reset_index_false() -> None:
    df1 = pd.DataFrame({"a": [1, 2]}, index=[10, 11])
    df2 = pd.DataFrame({"a": [3]}, index=[10])

    result = concat_dataframes([df1, df2], reset_index=False)

    assert list(result.index) == [10, 11, 10]


def test_concat_drops_empty_inputs() -> None:
    df1 = pd.DataFrame({"a": [1, 2]})
    empty = pd.DataFrame({"c": []})

    result = concat_dataframes([df1, empty])

    assert len(result) == 2
    assert list(result.columns) == ["a"]


def test_concat_all_empty_returns_empty_df() -> None:
    result = concat_dataframes([pd.DataFrame(), pd.DataFrame()])

    assert result.empty
    assert isinstance(result, pd.DataFrame)


def test_concat_all_empty_with_ensure_columns() -> None:
    result = concat_dataframes([pd.DataFrame()], ensure_columns=["a", "b"])

    assert result.empty
    assert list(result.columns) == ["a", "b"]


def test_concat_ensure_columns_adds_missing() -> None:
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})

    result = concat_dataframes([df1, df2], ensure_columns=["a", "missing"])

    assert "missing" in result.columns
    assert result["missing"].isna().all()
    assert list(result["a"]) == [1, 2]


def test_concat_preserves_geodataframe_and_crs() -> None:
    gdf1 = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    gdf2 = gpd.GeoDataFrame({"a": [2]}, geometry=[Point(1, 1)], crs="EPSG:4326")

    result = concat_dataframes([gdf1, gdf2])

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:4326"
    assert len(result) == 2


def test_concat_mixed_df_and_gdf_returns_gdf() -> None:
    df = pd.DataFrame({"a": [1]})
    gdf = gpd.GeoDataFrame({"a": [2]}, geometry=[Point(0, 0)], crs="EPSG:4326")

    result = concat_dataframes([df, gdf])

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:4326"
    assert len(result) == 2


def test_drop_skip_sentinels() -> None:
    df = pd.DataFrame({"a": [1]})

    assert _drop_skip_sentinels([SKIP_SENTINEL, df, SKIP_SENTINEL]) == [df]
    assert _drop_skip_sentinels([SKIP_SENTINEL]) == []


def test_concat_all_skipped_returns_empty_df() -> None:
    result = concat_dataframes(_drop_skip_sentinels([SKIP_SENTINEL, SKIP_SENTINEL]))

    assert result.empty
