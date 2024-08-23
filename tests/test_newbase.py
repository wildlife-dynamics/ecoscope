import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pandas.testing


def test_relocations_from_gdf():
    df = pd.read_feather("tests/sample_data/vector/movebank_data.feather")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.pop("location-long"), df.pop("location-lat")),
        crs=4326,
    )
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
    new = gdf.copy().relocations.from_gdf(
        groupby_col="individual-local-identifier",
        time_col="timestamp",
        uuid_col="event-id",
    )

    # assert on series.values instead of series because the index is different
    np.testing.assert_array_equal(new.gdf.index.to_series().values, gdf["event-id"].values)
    np.testing.assert_array_equal(new.gdf["fixtime"].values, gdf["timestamp"].values)
    np.testing.assert_array_equal(new.gdf["groupby_col"].values, gdf["individual-local-identifier"].values)
