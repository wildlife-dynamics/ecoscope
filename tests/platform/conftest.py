import geopandas as gpd
import pytest
from shapely.geometry import Point


@pytest.fixture
def spatial_feature_groups_table():
    import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
    import pyarrow as pa

    gdf = gpd.GeoDataFrame(
        {
            "pk": ["1", "2"],
            "name": ["Zone A", "Zone B"],
            "short_name": ["A", "B"],
            "feature_type": ["zone", "zone"],
            "metadata": [
                {"id": "g1", "display_name": "Group 1"},
                {"id": "g2", "display_name": "Group 2"},
            ],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    table = pa.table(
        {
            "pk": pa.array(gdf["pk"], type=pa.string()),
            "name": pa.array(gdf["name"], type=pa.string()),
            "short_name": pa.array(gdf["short_name"], type=pa.string()),
            "feature_type": pa.array(gdf["feature_type"], type=pa.string()),
            "metadata": pa.array(
                gdf["metadata"].tolist(),
                type=pa.struct([("id", pa.string()), ("display_name", pa.string())]),
            ),
            "geometry": ga.array(gdf.geometry.to_wkb().tolist()),
        }
    )
    return table
