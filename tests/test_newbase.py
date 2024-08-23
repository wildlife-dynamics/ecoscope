import pytest

import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import ecoscope


@pytest.fixture
def movebank_gdf():
    df = pd.read_feather("tests/sample_data/vector/movebank_data.feather")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.pop("location-long"), df.pop("location-lat")),
        crs=4326,
    )
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
    return gdf


def test_relocations_from_gdf(movebank_gdf):
    new = movebank_gdf.copy().relocations.from_gdf(
        groupby_col="individual-local-identifier",
        time_col="timestamp",
        uuid_col="event-id",
    )

    # assert on series.values instead of series because the index is different
    np.testing.assert_array_equal(new.index.to_series().values, movebank_gdf["event-id"].values)
    np.testing.assert_array_equal(new["fixtime"].values, movebank_gdf["timestamp"].values)
    np.testing.assert_array_equal(new["groupby_col"].values, movebank_gdf["individual-local-identifier"].values)
    assert not new["junk_status"].all()


def test_apply_recloc_filter(movebank_gdf):

    coord_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )

    new = (
        movebank_gdf.copy()
        .relocations.from_gdf(
            groupby_col="individual-local-identifier",
            time_col="timestamp",
            uuid_col="event-id",
        )
        .relocations.apply_reloc_filter(coord_filter)
    )
    assert new["junk_status"].any()


def test_traj_from_relocs(movebank_gdf):
    new = (
        movebank_gdf.copy()
        .relocations.from_gdf(
            groupby_col="individual-local-identifier",
            time_col="timestamp",
            uuid_col="event-id",
        )
        .trajectories.from_relocations()
    )
    assert "segment_start" in new.columns
