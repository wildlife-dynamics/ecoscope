"""Generate a synthetic example-return fixture for get_patrol_trajectory_from_smart.

No real SMART data is used. Features are fabricated to exercise the trajectory
schema the task emits (one trajectory per patrol, roster folded into list columns,
multiple leg-days, several mandates/stations). Run from the ecoscope repo:

    pixi run -e default python dev/gen_patrol_trajectory_fixture.py   # (any env with ecoscope)
"""

import geopandas as gpd
import pandas as pd
import shapely

from ecoscope.io.smartio import _PATROL_COLUMN_RENAMES, SmartIO
from ecoscope.io.utils import clean_time_cols
from ecoscope.trajectory import Trajectory

OUT = "ecoscope/platform/tasks/io/get-patrol-trajectory-from-smart.example-return.parquet"

# A few synthetic patrols, each spanning 1-2 leg-days, each leg-day carried by a
# 1-2 member roster (SMART repeats the track per member). Coordinates are made up.
SPECS = [
    # (patrol_uuid, serial, mandate, transport, station, [(legday, base_ms, lon0, [(leader, uuid)])])
    (
        "patrol-1",
        "MTri_0001",
        "Anti-Poaching",
        "Foot",
        "North Gate",
        [
            ("p1-day1", 1_700_000_000_000, 34.10, [("Ranger A", "u-a"), ("Ranger B", "u-b")]),
            ("p1-day2", 1_700_086_400_000, 34.30, [("Ranger A", "u-a"), ("Ranger B", "u-b")]),
        ],
    ),
    (
        "patrol-2",
        "MTri_0002",
        "Anti-Poaching",
        "Vehicle",
        "South Gate",
        [
            ("p2-day1", 1_700_100_000_000, 35.00, [("Ranger C", "u-c")]),
        ],
    ),
    (
        "patrol-3",
        "MTri_0003",
        "Rhino Monitoring",
        "Foot",
        "North Gate",
        [
            ("p3-day1", 1_700_200_000_000, 34.50, [("Ranger A", "u-a"), ("Ranger D", "u-d")]),
            ("p3-day2", 1_700_286_400_000, 34.70, [("Ranger A", "u-a"), ("Ranger D", "u-d")]),
        ],
    ),
    (
        "patrol-4",
        "MTri_0004",
        "Community Outreach",
        "Vehicle",
        "East Post",
        [
            ("p4-day1", 1_700_300_000_000, 35.20, [("Ranger E", "u-e")]),
        ],
    ),
]


def _track(base_ms, lon0):
    # five fixes ~1 min apart, drifting NE — Z is epoch-ms per from_track_geometry
    pts = [(lon0 + 0.001 * i, -1.000 - 0.001 * i, base_ms + 60_000 * i) for i in range(5)]
    return shapely.geometry.MultiLineString([shapely.geometry.LineString(pts)])


rows = []
for patrol_uuid, serial, mandate, transport, station, legdays in SPECS:
    for legday, base_ms, lon0, roster in legdays:
        for leader_name, leader_uuid in roster:  # SMART repeats track per member
            rows.append(
                {
                    "geometry": _track(base_ms, lon0),
                    "uuid": patrol_uuid,
                    "id": serial,
                    "patrol_leg_uuid": f"{legday}-leg",
                    "patrol_leg_id": f"{legday}-leg-id",
                    "patrol_leg_day_uuid": legday,
                    "patrol_leg_day_start": pd.Timestamp(base_ms, unit="ms", tz="UTC").isoformat(),
                    "patrol_leg_day_end": pd.Timestamp(base_ms + 240_000, unit="ms", tz="UTC").isoformat(),
                    "patrol_mandate": mandate,
                    "patrol_transport": transport,
                    "station": station,
                    "patrol_leader_name": leader_name,
                    "patrol_leader_uuid": leader_uuid,
                }
            )

raw = gpd.GeoDataFrame(rows, crs="EPSG:4326")

smart = SmartIO.__new__(SmartIO)  # bypass network login
patrols = smart._collapse_patrol_members(raw)
patrols = patrols.rename(columns=_PATROL_COLUMN_RENAMES)
patrols = smart._add_patrol_time_bounds(patrols)
patrols["patrol_type__display"] = patrols["patrol_mandate"]

traj = Trajectory.from_track_geometry(patrols, groupby_col="patrol_id")
traj.gdf.columns = [c.replace("extra__", "") for c in traj.gdf.columns]
traj.gdf = clean_time_cols(traj.gdf)

traj.gdf.to_parquet(OUT)
print("wrote", OUT)
print("rows", len(traj.gdf), "patrols", traj.gdf["groupby_col"].nunique())
print("cols", list(traj.gdf.columns))
print("leader sample", traj.gdf["patrol_leader_name"].iloc[0])
