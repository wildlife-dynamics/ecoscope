import logging

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.platform.tasks.transformation import (
    map_columns,
    reorder_columns,
    strip_prefix_from_column_names,
    title_case_columns_by_prefix,
)
from ecoscope.platform.tasks.transformation._mapping import RenameColumn


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
    return pd.DataFrame(data)


def test_drop_columns(sample_dataframe):
    """Test that columns are correctly dropped."""
    result_df = map_columns(sample_dataframe, drop_columns=["A"], retain_columns=[], rename_columns={})
    assert "A" not in result_df.columns


def test_drop_columns_error(sample_dataframe):
    """Test raising error if a column does not exist."""
    with pytest.raises(KeyError):
        map_columns(
            sample_dataframe,
            drop_columns=["NOT_EXIST"],
            retain_columns=[],
            rename_columns={},
        )


def test_drop_columns_ignore_missing(sample_dataframe):
    """Test that missing columns are ignored when raise_if_not_found is False."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=["A", "NOT_EXIST"],
        retain_columns=[],
        rename_columns={},
        raise_if_not_found=False,
    )
    assert "A" not in result_df.columns
    assert list(result_df.columns) == ["B", "C"]


def test_retain_columns(sample_dataframe):
    """Test that only specified columns are retained."""
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=["B"], rename_columns={})
    assert list(result_df.columns) == ["B"]


def test_retain_columns_respects_order(sample_dataframe):
    """Test that only specified columns are retained."""
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=["B", "A"], rename_columns={})
    assert list(result_df.columns) == ["B", "A"]


def test_retain_columns_error(sample_dataframe):
    """Test raising error if a column does not exist."""
    with pytest.raises(KeyError):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=["NOT_EXIST"],
            rename_columns={},
        )


def test_rename_columns(sample_dataframe):
    """Test that columns are correctly renamed."""
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns={"B": "Z"})
    assert "Z" in result_df.columns and "B" not in result_df.columns


def test_rename_columns_error(sample_dataframe):
    """Test raising error if a column does not exist."""
    with pytest.raises(KeyError, match=r"Columns \['NOT_EXIST'\] not all found in DataFrame\."):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=[],
            rename_columns={"NOT_EXIST": "Z"},
        )


def test_rename_columns_with_list(sample_dataframe):
    """Test that columns are correctly renamed using a list of RenameColumn objects."""
    rename_list = [
        RenameColumn(original_name="B", new_name="Z"),
        RenameColumn(original_name="C", new_name="Y"),
    ]
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns=rename_list)
    assert "Z" in result_df.columns and "B" not in result_df.columns
    assert "Y" in result_df.columns and "C" not in result_df.columns
    assert "A" in result_df.columns


def test_rename_columns_with_list_single_column(sample_dataframe):
    """Test renaming a single column using a list of RenameColumn objects."""
    rename_list = [RenameColumn(original_name="A", new_name="X")]
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns=rename_list)
    assert "X" in result_df.columns and "A" not in result_df.columns
    assert list(result_df.columns) == ["X", "B", "C"]


def test_rename_columns_with_list_error(sample_dataframe):
    """Test raising error if a column does not exist when using list format."""
    rename_list = [RenameColumn(original_name="NOT_EXIST", new_name="Z")]
    with pytest.raises(KeyError, match=r"Columns \['NOT_EXIST'\] not all found in DataFrame\."):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=[],
            rename_columns=rename_list,
        )


def test_map_columns_with_rename_list(sample_dataframe):
    """Test that columns are correctly mapped when using list format for renaming."""
    rename_list = [RenameColumn(original_name="B", new_name="Z")]
    result_df = map_columns(
        sample_dataframe,
        drop_columns=["C"],
        retain_columns=["B"],
        rename_columns=rename_list,
    )
    assert list(result_df.columns) == ["Z"]


def test_map_columns(sample_dataframe):
    """Test that columns are correctly mapped."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=["C"],
        retain_columns=["B"],
        rename_columns={"B": "Z"},
    )
    assert list(result_df.columns) == ["Z"]


def test_title_case_columns_by_prefix():
    df = pd.DataFrame(
        data={
            "a_value": [1, 2, 3],
            "extra__another_value": [4, 5, 6],
            "extra__a_third_value": [7, 8, 9],
        }
    )

    df = title_case_columns_by_prefix(df, prefix="extra__")
    assert df.columns.to_list() == [
        "a_value",
        "Another Value",
        "A Third Value",
    ]


def test_reorder_columns():
    df = pd.DataFrame(
        data={
            "a_value": [1, 2, 3],
            "another_value": [4, 5, 6],
            "a_third_value": [7, 8, 9],
        }
    )

    df = reorder_columns(df, columns=["a_third_value", "another_value"])
    assert df.columns.to_list() == [
        "a_third_value",
        "another_value",
        "a_value",
    ]


def test_rename_columns_skips_collision_by_default(sample_dataframe, caplog):
    """By default a rename onto an existing column is skipped, rather than creating duplicate labels."""
    with caplog.at_level(logging.WARNING):
        result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns={"A": "B"})

    assert "Not renaming columns whose new name is already taken: {'A': 'B'}" in caplog.text
    assert list(result_df.columns) == ["A", "B", "C"]
    assert result_df.columns.is_unique
    assert isinstance(result_df["B"], pd.Series)
    assert result_df["A"].to_list() == [1, 2, 3]
    assert result_df["B"].to_list() == [4, 5, 6]


def test_rename_columns_duplicate_overwrite(sample_dataframe):
    """Test replacing the existing column when its name is taken by a rename."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=[],
        retain_columns=[],
        rename_columns={"A": "B"},
        duplicate_strategy="overwrite",
    )
    assert list(result_df.columns) == ["B", "C"]
    assert result_df.columns.is_unique
    assert isinstance(result_df["B"], pd.Series)
    assert result_df["B"].to_list() == [1, 2, 3]


def test_rename_columns_duplicate_error(sample_dataframe):
    """Test raising if a new column name is already taken."""
    with pytest.raises(ValueError, match=r"would create duplicate columns: \['B'\]"):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=[],
            rename_columns={"A": "B"},
            duplicate_strategy="error",
        )


def test_rename_columns_duplicate_skip_cascades(sample_dataframe):
    """Skipping a rename keeps its original name taken, which can knock out a rename onto that name."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=[],
        retain_columns=[],
        # "C" -> "B" collides with "A" -> "B", so "C" stays put, which in turn collides with
        # "B" -> "C", so "B" stays put, which finally collides with "A" -> "B".
        rename_columns={"A": "B", "B": "C", "C": "B"},
        duplicate_strategy="skip",
    )
    assert list(result_df.columns) == ["A", "B", "C"]
    assert result_df["A"].to_list() == [1, 2, 3]
    assert result_df["B"].to_list() == [4, 5, 6]
    assert result_df["C"].to_list() == [7, 8, 9]


@pytest.mark.parametrize("duplicate_strategy", ["overwrite", "skip", "error"])
def test_rename_columns_swap_is_not_a_duplicate(sample_dataframe, duplicate_strategy):
    """Swapping two column names does not create duplicates, under any strategy."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=[],
        retain_columns=[],
        rename_columns={"A": "B", "B": "A"},
        duplicate_strategy=duplicate_strategy,
    )
    assert list(result_df.columns) == ["B", "A", "C"]
    assert result_df["B"].to_list() == [1, 2, 3]
    assert result_df["A"].to_list() == [4, 5, 6]


def test_rename_columns_shared_new_name_overwrite(sample_dataframe):
    """When two columns are renamed to the same name, the last one wins."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=[],
        retain_columns=[],
        rename_columns={"A": "Z", "B": "Z"},
        duplicate_strategy="overwrite",
    )
    assert list(result_df.columns) == ["Z", "C"]
    assert result_df.columns.is_unique
    assert result_df["Z"].to_list() == [4, 5, 6]


def test_rename_columns_shared_new_name_skip(sample_dataframe):
    """When two columns are renamed to the same name, the second rename is skipped."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=[],
        retain_columns=[],
        rename_columns={"A": "Z", "B": "Z"},
    )
    assert list(result_df.columns) == ["Z", "B", "C"]
    assert result_df["Z"].to_list() == [1, 2, 3]
    assert result_df["B"].to_list() == [4, 5, 6]


def test_rename_columns_shared_new_name_error(sample_dataframe):
    with pytest.raises(ValueError, match=r"would create duplicate columns: \['Z'\]"):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=[],
            rename_columns={"A": "Z", "B": "Z"},
            duplicate_strategy="error",
        )


def test_rename_columns_missing_column_is_not_a_duplicate(sample_dataframe):
    """A rename of a column that isn't present leaves the existing target untouched."""
    result_df = map_columns(
        sample_dataframe,
        drop_columns=[],
        retain_columns=[],
        rename_columns={"NOT_EXIST": "B"},
        raise_if_not_found=False,
    )
    assert list(result_df.columns) == ["A", "B", "C"]
    assert result_df["B"].to_list() == [4, 5, 6]


def test_rename_columns_with_list_skips_collision(sample_dataframe):
    """Duplicates are resolved the same way when renames are given as a list."""
    rename_list = [RenameColumn(original_name="A", new_name="B")]
    result_df = map_columns(sample_dataframe, drop_columns=[], retain_columns=[], rename_columns=rename_list)
    assert list(result_df.columns) == ["A", "B", "C"]
    assert result_df["A"].to_list() == [1, 2, 3]
    assert result_df["B"].to_list() == [4, 5, 6]


def test_rename_columns_invalid_duplicate_strategy(sample_dataframe):
    with pytest.raises(ValueError, match="Invalid selection for duplicate_strategy"):
        map_columns(
            sample_dataframe,
            drop_columns=[],
            retain_columns=[],
            rename_columns={"A": "B"},
            duplicate_strategy="not_a_strategy",  # type: ignore[arg-type]
        )


def test_rename_columns_renaming_a_duplicated_column_raises():
    """Renaming a duplicated label would rename every column holding it, so refuse to."""
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "A", "B"])

    with pytest.raises(ValueError, match=r"Cannot rename columns \['A'\]"):
        map_columns(df, drop_columns=[], retain_columns=[], rename_columns={"A": "Z"})


def test_rename_columns_leaves_untouched_duplicated_columns_alone():
    """Duplicate labels among the columns we aren't renaming are none of our business."""
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "A", "B"])

    result_df = map_columns(df, drop_columns=[], retain_columns=[], rename_columns={"B": "Z"})
    assert list(result_df.columns) == ["A", "A", "Z"]


@pytest.fixture
def sample_geodataframe():
    """A GeoDataFrame with a second, plain object column of geometries."""
    return gpd.GeoDataFrame(
        data={"A": [1, 2], "the_geom": [Point(0, 0), Point(1, 1)]},
        geometry=[Point(2, 2), Point(3, 3)],
        crs="EPSG:4326",
    )


@pytest.mark.parametrize("active_geometry", ["geometry", "geom"])
def test_rename_columns_overwriting_active_geometry_raises(sample_geodataframe, active_geometry):
    """Overwriting the active geometry column would demote the frame to a plain DataFrame."""
    gdf = sample_geodataframe
    if active_geometry != gdf.active_geometry_name:
        gdf = gdf.rename_geometry(active_geometry)

    with pytest.raises(ValueError, match=f"would overwrite the active geometry column '{active_geometry}'"):
        map_columns(
            gdf,
            drop_columns=[],
            retain_columns=[],
            rename_columns={"the_geom": active_geometry},
            duplicate_strategy="overwrite",
        )


def test_rename_columns_skips_active_geometry_by_default(sample_geodataframe):
    """The default strategy leaves the active geometry, and so the frame's spatial-ness, intact."""
    result_gdf = map_columns(
        sample_geodataframe, drop_columns=[], retain_columns=[], rename_columns={"the_geom": "geometry"}
    )

    assert isinstance(result_gdf, gpd.GeoDataFrame)
    assert result_gdf.active_geometry_name == "geometry"
    assert result_gdf.crs == "EPSG:4326"
    assert list(result_gdf.columns) == ["A", "the_geom", "geometry"]


def test_rename_columns_overwrite_preserves_geodataframe(sample_geodataframe):
    """Overwriting a column that isn't the geometry keeps the frame a GeoDataFrame, CRS and all."""
    result_gdf = map_columns(
        sample_geodataframe,
        drop_columns=[],
        retain_columns=[],
        rename_columns={"A": "the_geom"},
        duplicate_strategy="overwrite",
    )

    assert isinstance(result_gdf, gpd.GeoDataFrame)
    assert result_gdf.active_geometry_name == "geometry"
    assert result_gdf.crs == "EPSG:4326"
    assert list(result_gdf.columns) == ["the_geom", "geometry"]
    assert result_gdf["the_geom"].to_list() == [1, 2]


def test_rename_columns_geometry_can_be_replaced_by_dropping_it_first(sample_geodataframe):
    """Replacing the geometry is still possible, but it has to be asked for explicitly."""
    result_gdf = map_columns(
        sample_geodataframe, drop_columns=["geometry"], retain_columns=[], rename_columns={"the_geom": "geometry"}
    )

    assert list(result_gdf.columns) == ["A", "geometry"]
    assert result_gdf.columns.is_unique
    assert result_gdf["geometry"].to_list() == [Point(0, 0), Point(1, 1)]


def test_title_case_columns_by_prefix_collision():
    """A column whose title cased name is already taken keeps its original name."""
    df = pd.DataFrame(
        data={
            "Another Value": [1, 2, 3],
            "extra__another_value": [4, 5, 6],
        }
    )

    df = title_case_columns_by_prefix(df, prefix="extra__")
    assert df.columns.to_list() == ["Another Value", "extra__another_value"]
    assert df.columns.is_unique
    assert df["Another Value"].to_list() == [1, 2, 3]


def test_strip_prefix_from_column_names():
    df = pd.DataFrame(data={"extra__a": [1, 2, 3], "b": [4, 5, 6]})

    df = strip_prefix_from_column_names(df, prefix="extra__")
    assert df.columns.to_list() == ["a", "b"]


def test_strip_prefix_from_column_names_collision():
    """A column whose stripped name is already taken keeps its prefix."""
    df = pd.DataFrame(data={"extra__a": [1, 2, 3], "a": [4, 5, 6]})

    df = strip_prefix_from_column_names(df, prefix="extra__")
    assert df.columns.to_list() == ["extra__a", "a"]
    assert df.columns.is_unique
    assert df["a"].to_list() == [4, 5, 6]
