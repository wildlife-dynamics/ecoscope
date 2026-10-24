import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import ecoscope


def test_trajectory_keeps_list_and_dict_extra_columns():
    gdf = gpd.GeoDataFrame(
        {
            "groupby_col": ["p1"] * 3,
            "fixtime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10"], utc=True),
            "junk_status": [False] * 3,
            "patrol_segment_id": ["s1", "s2", "s3"],
            "patrol_members": [["a"], ["b"], ["c"]],
            "patrol_segment_details": [{"k": 1}, {"k": 2}, {"k": 3}],
        },
        geometry=[Point(36.0, 0.0), Point(36.001, 0.0), Point(36.002, 0.0)],
        crs=4326,
    )
    traj = ecoscope.Trajectory.from_relocations(ecoscope.Relocations(gdf=gdf))
    out = traj.gdf
    assert len(out) == 2
    assert out["extra__patrol_segment_id"].tolist() == ["s1", "s2"]
    assert out["extra__patrol_members"].tolist() == [["a"], ["b"]]
    assert out["extra__patrol_segment_details"].tolist() == [{"k": 1}, {"k": 2}]
