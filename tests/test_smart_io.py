import geopandas as gpd
import pandas as pd
import pytest
import shapely

from ecoscope.io.smartio import SmartIO
from ecoscope.io.utils import TIME_COLS

pytestmark = pytest.mark.smart_io


def check_time_is_parsed(df):
    for col in TIME_COLS:
        if col in df.columns:
            assert pd.api.types.is_datetime64_ns_dtype(df[col]) or df[col].isna().all()


@pytest.mark.skip(reason="SMART API is currently down")
def test_smart_get_events(smart_io):
    events = smart_io.get_events(
        ca_uuid="735606d2-c34e-49c3-a45b-7496ca834e58",
        language_uuid="13451893-86af-4ec0-beac-2b8e0c2482b5",
        start="2021-12-20",
        end="2021-12-22",
    )
    assert not events.empty
    assert "time" in events.columns
    assert "event_type" in events.columns
    assert "geometry" in events.columns
    assert "extracted_attributes" in events.columns
    check_time_is_parsed(events)


@pytest.mark.skip(reason="SMART API is currently down")
def test_smart_get_patrol_observations(smart_io):
    result = smart_io.get_patrol_observations(
        ca_uuid="735606d2-c34e-49c3-a45b-7496ca834e58",
        language_uuid="13451893-86af-4ec0-beac-2b8e0c2482b5",
        start="2021-12-20",
        end="2021-12-22",
    )

    assert len(result.gdf) > 0
    assert "geometry" in result.gdf
    assert "groupby_col" in result.gdf
    assert "fixtime" in result.gdf
    check_time_is_parsed(result.gdf)


def _patrol_feature(leader_name, leader_uuid):
    # MultiLineString Z vertices are (lon, lat, timestamp_ms)
    line = shapely.geometry.LineString(
        [
            (34.000, -1.000, 1_700_000_000_000),
            (34.001, -1.001, 1_700_000_060_000),
            (34.002, -1.002, 1_700_000_120_000),
        ]
    )
    return {
        "geometry": shapely.geometry.MultiLineString([line]),
        "uuid": "patrol-1",
        "id": "leg-1",
        "patrol_mandate": "Anti-Poaching",
        "patrol_transport": "Foot",
        "patrol_leader_name": leader_name,
        "patrol_leader_uuid": leader_uuid,
    }


def test_process_patrols_gdf_collapses_member_roster():
    # SMART returns the same track once per team member; expanding every feature would
    # otherwise multiply each fix by the team size.
    df = gpd.GeoDataFrame(
        [_patrol_feature("Alice", "u-a"), _patrol_feature("Bob", "u-b")],
        crs="EPSG:4326",
    )
    smart = SmartIO.__new__(SmartIO)  # bypass network login in __init__

    out = smart.process_patrols_gdf(df)

    # one row per physical fix (3 vertices), not 3 vertices x 2 members
    assert len(out) == 3
    # the full roster is preserved as a list on every fix
    leaders = out["patrol_leader_name"].iloc[0]
    assert isinstance(leaders, list)
    assert set(leaders) == {"Alice", "Bob"}
    assert set(out["patrol_leader_uuid"].iloc[0]) == {"u-a", "u-b"}
    # track attributes survive the collapse
    assert (out["groupby_col"] == "leg-1").all()
    assert (out["patrol_mandate"] == "Anti-Poaching").all()
    assert out["fixtime"].nunique() == 3
