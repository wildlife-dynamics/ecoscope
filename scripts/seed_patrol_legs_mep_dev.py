"""Idempotently seed mep-dev with patrols whose legs carry team, members, assets, details and a pause.

Reads EARTHRANGER_SERVER, EARTHRANGER_USERNAME, EARTHRANGER_PASSWORD. Never deletes anything.
The patrol configuration (team, member group, asset group) must already be set in the EarthRanger admin.
"""

import os
import sys
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from ecoscope.io import EarthRangerIO

TEAM_ID = "377a78fe-4677-4fca-b289-0e893ee08233"  # Ecoscope Leg Test Team
MEMBER_GROUP_ID = "ddb99202-1373-4c6c-b15f-e71d21cb2b26"  # Rangers
ASSET_GROUP_ID = "a3b3dc54-b094-4a0a-9b76-608bfab3fe9f"  # Ecoscope Leg Assets
PATROL_TYPE = "ecoscope_leg_test"
SUBJECTS = {
    "Ecoscope Leg Tester A": ("person", "ranger", MEMBER_GROUP_ID),
    "Ecoscope Leg Tester B": ("person", "ranger", MEMBER_GROUP_ID),
    "Ecoscope Leg Vehicle": ("vehicle", "suv", ASSET_GROUP_ID),
}
START_LON, START_LAT = 36.9, -1.4
LON_SPAN = 0.108  # ~12 km at this latitude

PATROLS = [
    {
        "title": "Ecoscope leg test 2026-01-05",
        "legs": [
            ("2026-01-05T08:00:00Z", "2026-01-05T12:00:00Z", False),
            ("2026-01-05T12:00:00Z", "2026-01-05T13:00:00Z", True),
            ("2026-01-05T13:00:00Z", "2026-01-05T16:00:00Z", False),
        ],
    },
    {
        "title": "Ecoscope leg test 2026-01-06",
        "legs": [("2026-01-06T09:00:00Z", "2026-01-06T11:00:00Z", False)],
    },
]


def as_list(resp):
    if isinstance(resp, dict):
        return resp.get("results", resp.get("data", []))
    return resp


def fail(exc):
    body = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
    print(f"BLOCKED: {exc}\n{body}")
    sys.exit(1)


def ensure_subject(io, name, subject_type, subtype, group_id):
    found = io.get_subjects(name=name)
    if len(found):
        subject_id = found["id"].iloc[0]
    else:
        subject_id = io._post(
            "subjects", payload={"name": name, "subject_type": subject_type, "subject_subtype": subtype}
        )["id"]
        print(f"created subject {name} {subject_id}")
    members = as_list(io._get(f"subjectgroup/{group_id}/subjects/"))
    if subject_id not in {m["id"] for m in members}:
        io._post(f"subjectgroup/{group_id}/subjects/", payload=[{"id": subject_id}])
        print(f"added {name} to group {group_id}")
    return subject_id


def ensure_source(io, subject_id, name):
    manufacturer_id = name.lower().replace(" ", "-")
    linked = [s for s in as_list(io._get(f"subject/{subject_id}/sources")) if s["manufacturer_id"] == manufacturer_id]
    if linked:
        return linked[0]["id"]
    created = io.post_source("tracking-device", manufacturer_id, "ecoscope-leg-test", provider="default")
    source_id = created["id"].iloc[0]
    print(f"created source {manufacturer_id} {source_id}")
    io.post_subjectsource(subject_id, source_id, "2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")
    print(f"linked source {source_id} to subject {subject_id}")
    return source_id


def ensure_observations(io, source_id, day):
    since, until = f"{day}T07:00:00Z", f"{day}T17:00:00Z"
    existing = io._get("observations", params={"source_id": source_id, "since": since, "until": until, "page_size": 1})
    if existing.get("count") if isinstance(existing, dict) else len(existing):
        return
    times = pd.date_range(f"{day}T08:00:00Z", f"{day}T16:00:00Z", freq="10min")
    lons = [START_LON + LON_SPAN * i / (len(times) - 1) for i in range(len(times))]
    obs = gpd.GeoDataFrame(
        {"source": source_id, "recorded_at": times.strftime("%Y-%m-%dT%H:%M:%SZ")},
        geometry=[Point(lon, START_LAT) for lon in lons],
        crs=4326,
    )
    io.post_observations(obs)
    print(f"posted {len(obs)} observations for {source_id} on {day}")


def ensure_patrol_type(io):
    types = io.get_patrol_types(include_inactive=True)
    if PATROL_TYPE not in set(types["value"]):
        io._post("activity/patrols/types/", payload={"value": PATROL_TYPE, "display": "Ecoscope Leg Test"})
        print(f"created patrol type {PATROL_TYPE}")


def ensure_patrol(io, spec, leader, members, assets):
    day = spec["legs"][0][0][:10]
    existing = io.get_patrols(
        since=f"{day}T00:00:00Z",
        until=f"{day}T23:59:59Z",
        patrol_type_value=[PATROL_TYPE],
        status=["scheduled", "active", "overdue", "done", "cancelled"],
    )
    if len(existing) and (existing["title"] == spec["title"]).any():
        row = existing[existing["title"] == spec["title"]].iloc[0]
        return row["id"] if "id" in row else row.name, row["serial_number"], False
    patrol = io.post_patrol(priority=0, title=spec["title"]).iloc[0]
    for start, end, is_pause in spec["legs"]:
        io.post_patrol_segment(
            patrol_id=patrol["id"],
            patrol_segment_id=str(uuid4()),
            patrol_type=PATROL_TYPE,
            tracked_subject_id=leader,
            start_time=start,
            end_time=end,
            team=TEAM_ID,
            members=members,
            assets=assets,
            segment_details={"weather": "dry"},
            type_details={},
            is_pause=is_pause,
        )
    io._patch(f"activity/patrols/{patrol['id']}/", payload={"state": "done"})
    return patrol["id"], patrol["serial_number"], True


def main():
    io = EarthRangerIO(
        server=os.environ["EARTHRANGER_SERVER"],
        username=os.environ["EARTHRANGER_USERNAME"],
        password=os.environ["EARTHRANGER_PASSWORD"],
    )
    try:
        ids = {name: ensure_subject(io, name, *cfg) for name, cfg in SUBJECTS.items()}
        for name, subject_id in ids.items():
            source_id = ensure_source(io, subject_id, name)
            if name == "Ecoscope Leg Tester A":
                for day in ("2026-01-05", "2026-01-06"):
                    ensure_observations(io, source_id, day)
        ensure_patrol_type(io)
        leader = ids["Ecoscope Leg Tester A"]
        members = [ids["Ecoscope Leg Tester A"], ids["Ecoscope Leg Tester B"]]
        assets = [ids["Ecoscope Leg Vehicle"]]
        serials = []
        for spec in PATROLS:
            patrol_id, serial, created = ensure_patrol(io, spec, leader, members, assets)
            print(f"patrol {spec['title']}: id={patrol_id} serial={serial} {'created' if created else 'exists'}")
            serials.append(int(serial))
    except requests.exceptions.RequestException as exc:
        fail(exc)
    print("serial numbers:", serials)


if __name__ == "__main__":
    main()
