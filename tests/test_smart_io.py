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
        "id": "MTri_0001",
        "patrol_leg_uuid": "leg-1",
        "patrol_leg_day_uuid": "legday-1",
        "patrol_mandate": "Anti-Poaching",
        "patrol_transport": "Foot",
        "patrol_leader_name": leader_name,
        "patrol_leader_uuid": leader_uuid,
    }


def test_collapse_patrol_members_folds_roster_per_leg_day():
    # SMART returns the same leg-day track once per team member; the duplicate features
    # collapse to one per leg-day before expansion, folding the roster into lists.
    df = gpd.GeoDataFrame(
        [_patrol_feature("Alice", "u-a"), _patrol_feature("Bob", "u-b")],
        crs="EPSG:4326",
    )
    smart = SmartIO.__new__(SmartIO)  # bypass network login in __init__

    out = smart._collapse_patrol_members(df)

    # two member features for one leg-day collapse to a single feature, track untouched
    assert len(out) == 1
    assert isinstance(out.geometry.iloc[0], shapely.geometry.MultiLineString)
    # the full roster is preserved as a list
    leaders = out["patrol_leader_name"].iloc[0]
    assert isinstance(leaders, list)
    assert set(leaders) == {"Alice", "Bob"}
    assert set(out["patrol_leader_uuid"].iloc[0]) == {"u-a", "u-b"}


def test_process_patrols_gdf_expands_collapsed_feature():
    # After collapsing, expansion yields one row per physical fix with the roster intact.
    df = gpd.GeoDataFrame(
        [_patrol_feature("Alice", "u-a"), _patrol_feature("Bob", "u-b")],
        crs="EPSG:4326",
    )
    smart = SmartIO.__new__(SmartIO)

    out = smart.process_patrols_gdf(smart._collapse_patrol_members(df))

    # one row per physical fix (3 vertices), not 3 vertices x 2 members
    assert len(out) == 3
    assert set(out["patrol_leader_name"].iloc[0]) == {"Alice", "Bob"}
    # track attributes survive the collapse
    assert (out["patrol_id"] == "patrol-1").all()
    # the SMART patrol serial is persisted under the ER-aligned column name
    assert (out["patrol_serial_number"] == "MTri_0001").all()
    assert (out["patrol_mandate"] == "Anti-Poaching").all()
    assert out["fixtime"].nunique() == 3


def _feature(patrol_uuid, serial, leg_day_uuid, leader_name, leader_uuid, base_time, lon0=34.000):
    # MultiLineString Z vertices are (lon, lat, timestamp_ms); offset the track per leg-day
    line = shapely.geometry.LineString(
        [
            (lon0 + 0.000, -1.000, base_time),
            (lon0 + 0.001, -1.001, base_time + 60_000),
            (lon0 + 0.002, -1.002, base_time + 120_000),
        ]
    )
    return {
        "geometry": shapely.geometry.MultiLineString([line]),
        "uuid": patrol_uuid,
        "id": serial,
        "patrol_leg_uuid": f"{leg_day_uuid}-leg",
        "patrol_leg_day_uuid": leg_day_uuid,
        "patrol_leg_day_start": pd.Timestamp(base_time, unit="ms", tz="UTC").isoformat(),
        "patrol_leg_day_end": pd.Timestamp(base_time + 120_000, unit="ms", tz="UTC").isoformat(),
        "patrol_mandate": "Anti-Poaching",
        "patrol_transport": "Foot",
        "patrol_leader_name": leader_name,
        "patrol_leader_uuid": leader_uuid,
    }


def test_one_trajectory_per_patrol_across_leg_days():
    from ecoscope.relocations import Relocations

    # Patrol A: 2 leg-days, each carried by 2 members (4 features); Patrol B: 1 leg-day, 1 member.
    features = [
        _feature("patrol-A", "MTri_A", "A-day1", "Alice", "u-a", 1_700_000_000_000),
        _feature("patrol-A", "MTri_A", "A-day1", "Bob", "u-b", 1_700_000_000_000),
        _feature("patrol-A", "MTri_A", "A-day2", "Alice", "u-a", 1_700_100_000_000),
        _feature("patrol-A", "MTri_A", "A-day2", "Bob", "u-b", 1_700_100_000_000),
        _feature("patrol-B", "MTri_B", "B-day1", "Carol", "u-c", 1_700_200_000_000),
    ]
    df = gpd.GeoDataFrame(features, crs="EPSG:4326")
    smart = SmartIO.__new__(SmartIO)

    out = smart.process_patrols_gdf(smart._collapse_patrol_members(df))

    # members collapsed per leg-day: A=2 leg-days x 3 fixes, B=1 leg-day x 3 fixes => 9 rows (not 15)
    assert len(out) == 9

    relocs = Relocations.from_gdf(out, groupby_col="patrol_id", uuid_col="patrol_id", time_col="fixtime")
    # exactly one trajectory group per patrol; A's two leg-days (6 fixes) form a single group
    assert relocs.gdf["groupby_col"].nunique() == 2
    assert (relocs.gdf["groupby_col"] == "patrol-A").sum() == 6
    assert (relocs.gdf["groupby_col"] == "patrol-B").sum() == 3


def test_from_track_geometry_no_cross_track_segments():
    from ecoscope.trajectory import Trajectory

    # two spatially distant single-LineString tracks under one trajectory id
    line1 = shapely.geometry.LineString([(34.00, -1.0, 1_700_000_000_000), (34.01, -1.0, 1_700_000_060_000)])
    line2 = shapely.geometry.LineString([(40.00, -1.0, 1_700_100_000_000), (40.01, -1.0, 1_700_100_060_000)])
    gdf = gpd.GeoDataFrame(
        {"pid": ["P", "P"]},
        geometry=[shapely.geometry.MultiLineString([line1]), shapely.geometry.MultiLineString([line2])],
        crs="EPSG:4326",
    )

    traj = Trajectory.from_track_geometry(gdf, groupby_col="pid")

    # each track -> 1 segment; one group, NO bridging segment between the tracks => 2 (not 3)
    assert len(traj.gdf) == 2
    assert (traj.gdf["groupby_col"] == "P").all()
    # the ~650km gap between tracks is never turned into a segment
    assert traj.gdf["dist_meters"].max() < 10_000


def test_get_patrol_trajectory_no_bridge_across_leg_days(monkeypatch):
    from ecoscope.trajectory import Trajectory

    # Patrol A: 2 spatially distant leg-days, each carried by 2 members (4 raw features)
    features = [
        _feature("patrol-A", "MTri_A", "A-day1", "Alice", "u-a", 1_700_000_000_000, lon0=34.0),
        _feature("patrol-A", "MTri_A", "A-day1", "Bob", "u-b", 1_700_000_000_000, lon0=34.0),
        _feature("patrol-A", "MTri_A", "A-day2", "Alice", "u-a", 1_700_100_000_000, lon0=40.0),
        _feature("patrol-A", "MTri_A", "A-day2", "Bob", "u-b", 1_700_100_000_000, lon0=40.0),
    ]
    raw = gpd.GeoDataFrame(features, crs="EPSG:4326")
    smart = SmartIO.__new__(SmartIO)
    monkeypatch.setattr(smart, "get_patrols_list", lambda **kwargs: raw)

    traj = smart.get_patrol_trajectory(ca_uuid="x", language_uuid="y", start="2026-04-01", end="2026-04-02")

    assert isinstance(traj, Trajectory)
    # 2 leg-days x (3 vertices -> 2 segments) = 4 segments; a bridge would make it 5
    assert len(traj.gdf) == 4
    # a single trajectory for the patrol
    assert (traj.gdf["groupby_col"] == "patrol-A").all()
    # no phantom segment spanning the gap between the two leg-days
    assert traj.gdf["dist_meters"].max() < 10_000
    # roster folded and serial persisted onto every segment
    assert set(traj.gdf["patrol_leader_name"].iloc[0]) == {"Alice", "Bob"}
    assert (traj.gdf["patrol_serial_number"] == "MTri_A").all()
    # patrol start/end span both leg-days: earliest leg-day start, latest leg-day end
    assert (traj.gdf["patrol_start_time"] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")).all()
    assert (traj.gdf["patrol_end_time"] == pd.Timestamp(1_700_100_000_000 + 120_000, unit="ms", tz="UTC")).all()
