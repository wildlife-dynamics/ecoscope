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
