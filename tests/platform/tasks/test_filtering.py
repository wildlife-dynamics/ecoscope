import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from ecoscope.platform.tasks.transformation._filtering import (
    BoundingBox,
    Coordinate,
    apply_reloc_coord_filter,
    drop_nan_values_by_column,
)
from shapely.geometry import Point, Polygon


@pytest.fixture
def df_with_geometry():
    data = {
        "geometry": [
            Point(0.0, 0.0),
            Point(100.0, 50.0),
            Point(-170.0, 80.0),
            Point(20.0, -20.0),
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_with_null_geometry():
    data = {
        "geometry": [
            Point(0.0, 0.0),
            None,
            Point(20.0, -20.0),
        ]
    }
    return pd.DataFrame(data)


def test_filter_points(df_with_geometry):
    # Sample data fixture (can be replaced with a parametrized fixture)
    expected_df = pd.DataFrame(
        {
            "geometry": [
                Point(100.0, 50.0),
                Point(-170.0, 80.0),
                Point(20.0, -20.0),
            ]
        }
    )

    # Apply the filter
    filtered_df = apply_reloc_coord_filter(df_with_geometry, filter_point_coords=[Coordinate(x=0.0, y=0.0)])

    # Assert that the filtered DataFrame matches the expected result
    pd.testing.assert_frame_equal(filtered_df, expected_df)


def test_filter_points_preserves_null_geometry(df_with_null_geometry):
    # Sample data fixture (can be replaced with a parametrized fixture)
    expected_df = pd.DataFrame(
        {
            "geometry": [
                None,
                Point(20.0, -20.0),
            ]
        }
    )

    # Apply the filter
    filtered_df = apply_reloc_coord_filter(df_with_null_geometry, filter_point_coords=[Coordinate(x=0.0, y=0.0)])

    # Assert that the filtered DataFrame matches the expected result
    pd.testing.assert_frame_equal(filtered_df, expected_df)


def test_filter_range(df_with_geometry):
    # Sample data fixture (can be replaced with a parametrized fixture)
    expected_df = pd.DataFrame({"geometry": [Point(-170.0, 80.0)]})

    # Apply the filter
    filtered_df = apply_reloc_coord_filter(
        df_with_geometry,
        BoundingBox(max_x=0),
    )

    # Assert that the filtered DataFrame matches the expected result
    pd.testing.assert_frame_equal(filtered_df, expected_df)


def test_filter_range_preserves_null_geometry(df_with_geometry):
    # Sample data fixture (can be replaced with a parametrized fixture)
    expected_df = pd.DataFrame({"geometry": [Point(-170.0, 80.0)]})

    # Apply the filter
    filtered_df = apply_reloc_coord_filter(
        df_with_geometry,
        BoundingBox(max_x=0),
    )

    # Assert that the filtered DataFrame matches the expected result
    pd.testing.assert_frame_equal(filtered_df, expected_df)


def test_filter_with_roi(df_with_geometry):
    expected_df = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(100, 50)]})

    gdf = gpd.GeoDataFrame(df_with_geometry, geometry="geometry")
    roi_gdf = gpd.GeoDataFrame(
        {
            "name": ["roi1"],
            "geometry": [Polygon([(0, 0), (150, 0), (150, 60), (0, 60), (0, 0)])],
        }
    )
    roi_gdf.set_crs(4326, inplace=True)
    roi_gdf.set_index("name", inplace=True)

    filtered_df = apply_reloc_coord_filter(gdf, roi_gdf=roi_gdf, roi_name="roi1")

    pd.testing.assert_frame_equal(filtered_df, expected_df)


def test_drop_nan_values_by_column():
    df_with_nans = pd.DataFrame(
        {"data": [np.nan, 14.0, np.nan, 16.0]},
        index=[0, 1, 2, 3],
    )

    df_without_nans = pd.DataFrame(
        {"data": [14.0, 16.0]},
        index=[1, 3],
    )
    pd.testing.assert_frame_equal(drop_nan_values_by_column(df_with_nans, "data"), df_without_nans)
