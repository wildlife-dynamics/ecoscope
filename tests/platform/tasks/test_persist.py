import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
from shapely.geometry import Point

from ecoscope.platform.tasks.io import persist_df


def test_persist_df_auto_filename_hashable(tmp_path):
    """Test automatic filename generation with hashable data."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")

    # Should generate a filename automatically
    dst = persist_df(df, root_path, None, "csv")

    # Verify file was created and contains correct data
    df_read = pd.read_csv(dst, index_col=0)
    pd.testing.assert_frame_equal(df_read, df)

    # Verify same dataframe generates same filename (deterministic)
    dst2 = persist_df(df, root_path, None, "csv")
    assert dst == dst2


def test_persist_df_auto_filename_unhashable(tmp_path):
    """Test automatic filename generation with unhashable data (fallback path)."""
    # Create a dataframe with unhashable types (e.g., lists)
    df = pd.DataFrame({"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]})
    root_path = str(tmp_path / "test")

    # Should generate a filename using the fallback method
    dst = persist_df(df, root_path, None, "csv")

    # Verify file was created
    df_read = pd.read_csv(dst, index_col=0)
    # Note: lists are stored as strings in CSV, so we can't directly compare
    assert len(df_read) == len(df)

    # Verify same dataframe generates same filename (deterministic)
    dst2 = persist_df(df, root_path, None, "csv")
    assert dst == dst2


def test_persist_df_auto_filename_different_data(tmp_path):
    """Test that different dataframes generate different filenames."""
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
    root_path = str(tmp_path / "test")

    dst1 = persist_df(df1, root_path, None, "csv")
    dst2 = persist_df(df2, root_path, None, "csv")

    # Different data should generate different filenames
    assert dst1 != dst2


def test_persist_df_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")
    filename = "data"
    dst = persist_df(df, root_path, filename, "csv")
    df_read = pd.read_csv(dst, index_col=0)
    pd.testing.assert_frame_equal(df_read, df)


def test_persist_df_gpkg(tmp_path):
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "geometry": [
                Point(0, 0),
                Point(1, 1),
                Point(2, 2),
            ],
        }
    )
    root_path = str(tmp_path / "test")
    filename = "data"
    dst = persist_df(df, root_path, filename, "gpkg")

    gdf_read = gpd.read_file(dst)
    pd.testing.assert_frame_equal(gdf_read, gpd.GeoDataFrame(df))
