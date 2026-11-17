import logging
from importlib.resources import files
from unittest.mock import MagicMock

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.io import EarthRangerIO
from ecoscope.platform.connections import EarthRangerBackends
from ecoscope.platform.mock_loaders import load_parquet
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import _incidents, get_incident_events

SVG = "<svg viewBox='0 0 10 10'><circle r='5'/></svg>"
GENERIC_SVG = "<svg><title>generic-gray</title><circle r='5'/></svg>"

SPEC_COLUMNS = [
    "incident_id",
    "incident",
    "incident_serial",
    "incident_title",
    "incident_state",
    "incident_time",
    "incident_created_at",
    "incident_updated_at",
    "incident_priority",
    "id",
    "serial_number",
    "time",
    "event_type",
    "event_type_display",
    "event_category",
    "event_category_display",
    "title",
    "state",
    "priority",
    "reported_by",
    "notes",
    "created_at",
    "updated_at",
    "event_details",
    "event_details_raw",
    "image_url",
    "icon",
    "geometry",
]


def _event(id, serial, time="2025-03-02T10:00:00Z", event_type="snare", contains=(), collection=False, **kw):
    return {
        "id": id,
        "serial_number": serial,
        "time": time,
        "event_type": event_type,
        "event_category": "security",
        "title": kw.get("title", f"Event {serial}"),
        "state": "active",
        "priority": 0,
        "reported_by": {"name": "Ranger A"},
        "notes": [{"text": "a note"}],
        "created_at": time,
        "updated_at": time,
        "event_details": {},
        "image_url": kw.get("image_url", f"https://er.example/{event_type}.svg"),
        "is_collection": collection,
        "contains": [{"type": "contains", "related_event": {"id": c}} for c in contains],
        "geometry": kw.get("geometry", Point(36.8, -1.3)),
    }


def _client(records, incident_ids):
    by_id = {r["id"]: r for r in records}

    def frame(rows):
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else pd.DataFrame()

    def get_events(**kw):
        if kw.get("is_collection"):
            return frame([by_id[i] for i in incident_ids])
        return frame([by_id[i] for i in kw.get("event_ids", []) if i in by_id])

    api = MagicMock(spec=EarthRangerIO)
    api.get_events.side_effect = get_events
    api.get_event_type_display_names_from_events.side_effect = lambda df, **kw: df.assign(
        event_type_display=df["event_type"].str.title()
    )
    api.get_event_categories.return_value = [{"value": "security", "display": "Security"}]
    return EarthRangerBackends(api=api, warehouse=None)


TIME_RANGE = TimeRange(since="2025-03-01T00:00:00Z", until="2025-03-31T00:00:00Z", timezone=UTC_TIMEZONEINFO)


@pytest.fixture(autouse=True)
def _no_schema_lookups(monkeypatch):
    monkeypatch.setattr(_incidents, "process_events_details", lambda df, client, **kw: df)
    _incidents._ICON_CACHE.clear()


def _run(records, incident_ids, **kw):
    return get_incident_events(client=_client(records, incident_ids), time_range=TIME_RANGE, **kw)


def test_members_fetched_by_id_and_labelled():
    records = [
        _event("A", 7, contains=["e1", "e2"], collection=True, event_type="incident_collection", title="Snaring"),
        _event("e1", 8),
        _event("e2", 9, time="2024-12-01T00:00:00Z"),  # outside the incident's range
    ]
    df = _run(records, ["A"])

    assert set(df["id"]) == {"e1", "e2"}
    assert set(df["incident"]) == {"#7 Snaring"}
    assert set(df["incident_serial"]) == {7}
    assert set(df["event_type_display"]) == {"Snare"}
    assert set(df["event_category_display"]) == {"Security"}
    assert set(df["reported_by"]) == {"Ranger A"}
    assert set(df["notes"]) == {"a note"}
    assert str(df["time"].dtype).startswith("datetime64")


def test_output_has_spec_columns():
    df = _run([_event("A", 7, contains=["e1"], collection=True), _event("e1", 8)], ["A"])

    assert [c for c in SPEC_COLUMNS if c not in df.columns] == []


def test_nested_incident_members_go_to_top_level():
    records = [
        _event("A", 1, contains=["B"], collection=True),
        _event("B", 2, contains=["e3"], collection=True),
        _event("e3", 3),
    ]
    df = _run(records, ["A"])

    assert list(df["id"]) == ["e3"]
    assert list(df["incident_id"]) == ["A"]


def test_max_depth_stops_walk():
    records = [
        _event("A", 1, contains=["B"], collection=True),
        _event("B", 2, contains=["e3"], collection=True),
        _event("e3", 3),
    ]

    assert _run(records, ["A"], max_depth=1).empty


def test_member_in_two_incidents_gives_two_rows():
    records = [
        _event("A", 1, contains=["e1"], collection=True),
        _event("C", 2, contains=["e1"], collection=True),
        _event("e1", 3),
    ]
    df = _run(records, ["A", "C"])

    assert sorted(df["incident_id"]) == ["A", "C"]
    assert set(df["id"]) == {"e1"}


def test_incident_without_members_is_skipped():
    records = [_event("A", 1, contains=["e1"], collection=True), _event("C", 2, collection=True), _event("e1", 3)]
    df = _run(records, ["A", "C"])

    assert set(df["incident_id"]) == {"A"}


def test_missing_members_warn(caplog):
    records = [_event("A", 1, contains=["e1", "e9"], collection=True), _event("e1", 2)]
    with caplog.at_level(logging.WARNING):
        df = _run(records, ["A"])

    assert list(df["id"]) == ["e1"]
    assert "1 member events" in caplog.text


def test_icons_off_by_default(monkeypatch):
    get = MagicMock()
    monkeypatch.setattr(_incidents.requests, "get", get)
    df = _run([_event("A", 1, contains=["e1"], collection=True), _event("e1", 2)], ["A"])

    get.assert_not_called()
    assert list(df["icon"]) == [""]


def test_icons_embedded_and_generic_dropped(monkeypatch):
    def get(url, timeout):
        if "broken" in url:
            raise OSError("down")
        return MagicMock(text=GENERIC_SVG if "generic" in url else SVG)

    monkeypatch.setattr(_incidents.requests, "get", get)
    records = [
        _event("A", 1, contains=["e1", "e2", "e3"], collection=True),
        _event("e1", 2, image_url="https://er.example/snare.svg"),
        _event("e2", 3, event_type="fire", image_url="https://er.example/generic.svg"),
        _event("e3", 4, event_type="gun", image_url="https://er.example/broken.svg"),
    ]
    icons = dict(zip(*_run(records, ["A"], include_icons=True)[["id", "icon"]].T.values))

    assert icons["e1"].startswith("data:image/svg+xml;base64,")
    assert icons["e2"] == ""
    assert icons["e3"] == ""


def test_no_incidents_returns_empty_frame():
    df = _run([], [])

    assert len(df) == 0
    assert {"id", "incident", "geometry"} <= set(df.columns)


def test_example_return_matches_schema():
    df = load_parquet(files("ecoscope.platform.tasks.io") / "get-incident-events.example-return.parquet")

    assert [c for c in SPEC_COLUMNS if c not in df.columns] == []
    assert df["incident_id"].nunique() >= 2
    assert df.geometry.isna().any() or df.geometry.is_empty.any()
    assert df["reported_by"].str.startswith("Ranger ").all()
