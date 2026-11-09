import geopandas as gpd
import pandas as pd
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


@pytest.fixture
def patrol_events_gdf():
    return gpd.GeoDataFrame(
        {
            "event_type": ["arrest", "sighting", "snare"],
            "time": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 09:30", "2025-01-02 10:00"]).tz_localize(
                "Africa/Nairobi"
            ),
            "patrol_id": ["p1", "p1", "p2"],
        },
        geometry=[Point(36.8, -1.3), Point(36.9, -1.4), Point(37.0, -1.5)],
        crs="EPSG:4326",
        index=[10, 11, 12],
    )


@pytest.fixture
def patrol_legs_df():
    start = pd.to_datetime(["2025-01-01 08:00", "2025-01-02 08:00"]).tz_localize("UTC")
    return pd.DataFrame(
        {
            "patrol_id": ["p1", "p2"],
            "patrol_serial_number": ["101", "102"],
            "patrol_segment_id": ["s1", "s2"],
            "patrol_type": ["foot", "vehicle"],
            "patrol_subject": ["Ranger A", "Ranger B"],
            "segment_start": start,
            "segment_end": start + pd.Timedelta(4, unit="h"),
            "leg_active_seconds": [14400.0, 14000.0],
            "leg_member_days": [0.5, 0.4],
            "patrol_elapsed_seconds": [14400.0, 14400.0],
        }
    )
