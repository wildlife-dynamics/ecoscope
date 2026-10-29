import numpy as np
import pandas as pd

from ecoscope.platform.indexes import AllGrouper, SpatialGrouper, ValueGrouper
from ecoscope.platform.tasks.transformation._leg_apportion import apportion_legs_by_spatial_index

legs = pd.DataFrame(
    {
        "patrol_segment_id": ["s1", "s2"],
        "leg_active_seconds": [7200.0, 3600.0],
        "leg_member_days": [2.0, 1.0],
        "patrol_elapsed_seconds": [7200.0, np.nan],
    }
)
indexed_traj = pd.DataFrame(
    {
        "patrol_segment_id": ["s1", "s1", "s1"],
        "dist_meters": [300.0, 100.0, 0.0],
        "SpatialGrouper_regions": ["North", "South", "South"],
    }
).set_index("SpatialGrouper_regions", append=True)
groupers = [SpatialGrouper(spatial_index_name="regions")]


def test_apportion_splits_by_track_share():
    out = apportion_legs_by_spatial_index(legs, indexed_traj, groupers).reset_index()
    s1 = out[out["patrol_segment_id"] == "s1"].set_index("SpatialGrouper_regions")
    assert s1.loc["North", "leg_active_seconds"] == 5400.0 and s1.loc["South", "leg_active_seconds"] == 1800.0
    assert s1.loc["North", "leg_member_days"] == 1.5 and s1.loc["North", "patrol_elapsed_seconds"] == 5400.0
    assert out["leg_active_seconds"].sum() == 7200.0 + 3600.0


def test_apportion_leg_without_track_lands_in_none_group_whole():
    out = apportion_legs_by_spatial_index(legs, indexed_traj, groupers).reset_index()
    s2 = out[out["patrol_segment_id"] == "s2"]
    assert s2["SpatialGrouper_regions"].tolist() == ["None"] and s2["leg_active_seconds"].iloc[0] == 3600.0


def test_apportion_noop_without_spatial_grouper():
    assert apportion_legs_by_spatial_index(legs, indexed_traj, [ValueGrouper(index_name="patrol_type")]).equals(legs)
    assert apportion_legs_by_spatial_index(legs, indexed_traj, AllGrouper()).equals(legs)


def test_apportion_leg_count_counts_once_per_feature():
    out = apportion_legs_by_spatial_index(legs, indexed_traj, groupers)
    assert out.groupby(level="SpatialGrouper_regions")["patrol_segment_id"].nunique().to_dict() == {
        "North": 1,
        "South": 1,
        "None": 1,
    }


def test_apportion_second_pass_with_same_grouper_is_idempotent():
    once = apportion_legs_by_spatial_index(legs, indexed_traj, groupers)
    twice = apportion_legs_by_spatial_index(once, indexed_traj, groupers)
    assert twice.equals(once)
    assert twice["leg_active_seconds"].sum() == 7200.0 + 3600.0


def test_apportion_tracks_without_segment_id_put_every_leg_in_none_group():
    traj_no_id = indexed_traj.drop(columns="patrol_segment_id")
    out = apportion_legs_by_spatial_index(legs, traj_no_id, groupers).reset_index()
    assert out["SpatialGrouper_regions"].tolist() == ["None", "None"]
    assert out["leg_active_seconds"].tolist() == [7200.0, 3600.0]


def test_apportion_drop_unplaced_removes_legs_without_a_placeable_track():
    out = apportion_legs_by_spatial_index(legs, indexed_traj, groupers, drop_unplaced=True).reset_index()
    assert out["patrol_segment_id"].tolist() == ["s1", "s1"]
    assert sorted(out["SpatialGrouper_regions"]) == ["North", "South"]
    assert out["leg_active_seconds"].sum() == 7200.0

    traj_no_id = indexed_traj.drop(columns="patrol_segment_id")
    none_left = apportion_legs_by_spatial_index(legs, traj_no_id, groupers, drop_unplaced=True)
    assert none_left.empty and "SpatialGrouper_regions" in none_left.index.names


def test_apportion_empty_leg_frame_passes_through():
    for empty in (pd.DataFrame(), legs.iloc[0:0]):
        out = apportion_legs_by_spatial_index(empty, indexed_traj, groupers)
        assert out.empty and out is empty


def test_apportion_spreads_patrol_elapsed_by_whole_patrol_track_share():
    # leg A (2 h, all in North) carries the patrol's 10 h elapsed; leg B (6 h) is all in South
    patrol_legs = pd.DataFrame(
        {
            "patrol_id": ["p1", "p1", "p2"],
            "patrol_segment_id": ["a", "b", "c"],
            "leg_active_seconds": [7200.0, 21600.0, 3600.0],
            "patrol_elapsed_seconds": [36000.0, np.nan, 3600.0],
        }
    )
    traj = pd.DataFrame(
        {
            "patrol_segment_id": ["a", "b", "b"],
            "dist_meters": [100.0, 200.0, 100.0],
            "SpatialGrouper_regions": ["North", "South", "North"],
        }
    ).set_index("SpatialGrouper_regions", append=True)
    out = apportion_legs_by_spatial_index(patrol_legs, traj, groupers)
    by_region = out.groupby(level="SpatialGrouper_regions")[["leg_active_seconds", "patrol_elapsed_seconds"]].sum()
    # p1 track: 200 m North (a:100 + b:100), 200 m South -> elapsed split 50/50; p2 has no track -> "None" whole
    assert by_region.loc["North", "patrol_elapsed_seconds"] == 18000.0
    assert by_region.loc["South", "patrol_elapsed_seconds"] == 18000.0
    assert by_region.loc["None", "patrol_elapsed_seconds"] == 3600.0
    assert by_region.loc["North", "leg_active_seconds"] == 7200.0 + 21600.0 / 3
    assert out["patrol_elapsed_seconds"].sum() == 36000.0 + 3600.0
    assert out["leg_active_seconds"].sum() == patrol_legs["leg_active_seconds"].sum()
