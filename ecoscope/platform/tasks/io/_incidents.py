"""EarthRanger incidents with their member events.

An incident is an EarthRanger event whose type has ``is_collection = true``; its members
are linked by ``contains`` relationships. The events API only returns those links with
``include_related_events=true``, and each link carries just a short header, so member
events are fetched again by id (they can be dated outside the incident's own range).
Incidents have no location of their own; their members carry it.
"""

import base64
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, cast

import pandas as pd
import requests  # type: ignore[import-untyped]
from erclient.client import ERClientException  # type: ignore[import-untyped]
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.connections import EarthRangerSource, as_backends
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope.platform.tasks.io._earthranger import process_events_details

logger = logging.getLogger(__name__)

ID_CHUNK = 50  # event ids per request: long id lists are not split for us
WORKERS = 5
ICON_PX = 64
_ICON_CACHE: dict[str, str] = {}

INCIDENT_COLUMNS = {
    "id": "incident_id",
    "incident": "incident",
    "serial_number": "incident_serial",
    "title": "incident_title",
    "state": "incident_state",
    "time": "incident_time",
    "created_at": "incident_created_at",
    "updated_at": "incident_updated_at",
    "priority": "incident_priority",
}
MEMBER_COLUMNS = [
    "id",
    "serial_number",
    "time",
    "event_type",
    "event_category",
    "title",
    "state",
    "priority",
    "priority_label",
    "reported_by",
    "notes",
    "created_at",
    "updated_at",
    "event_details",
    "image_url",
    "geometry",
]


def _events(api: Any, **kwargs: Any) -> pd.DataFrame:
    df = api.get_events(drop_null_geometry=False, include_details=True, include_notes=True, **kwargs)
    return df.reset_index() if "id" not in df.columns else df


def _events_by_ids(api: Any, ids: list[str]) -> pd.DataFrame:
    chunks = [ids[i : i + ID_CHUNK] for i in range(0, len(ids), ID_CHUNK)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        parts = [
            p for p in pool.map(lambda c: _events(api, event_ids=c, include_related_events=True), chunks) if len(p)
        ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id"])


def _contains(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(incident id, member id) for every `contains` relationship."""
    if "contains" not in df.columns:
        return []
    return [
        (inc_id, r["related_event"]["id"])
        for inc_id, rels in zip(df["id"], df["contains"])
        for r in (rels if isinstance(rels, list) else [])
        if isinstance(r, dict) and r.get("type") == "contains" and r.get("related_event")
    ]


def _is_collection(df: pd.DataFrame) -> pd.Series:
    return df.get("is_collection", pd.Series(False, index=df.index)).fillna(False).astype(bool)


def _icon_data_uri(url: Any) -> str:
    """The event type's SVG as a data URI with an explicit size (EarthRanger's SVGs often have
    none), or "" when there is none, it can't be fetched, or it is EarthRanger's generic icon
    (a plain disc given to types without their own), so consumers draw their own marker."""
    if not isinstance(url, str) or not url:
        return ""
    if url not in _ICON_CACHE:
        try:
            svg = requests.get(url, timeout=20).text
            svg = svg[svg.find("<svg") :]
            end = svg.find(">")
            head = re.sub(r"""\s(width|height)=(["']).*?\2""", "", svg[:end])
            svg = head.replace("<svg", f'<svg width="{ICON_PX}" height="{ICON_PX}"', 1) + svg[end:]
            generic = "<title>generic" in svg
            _ICON_CACHE[url] = "" if generic else "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
        except (requests.RequestException, OSError, ValueError):  # an icon is decoration: never fail over it
            logger.warning("Could not fetch event-type icon %s", url)
            _ICON_CACHE[url] = ""
    return _ICON_CACHE[url]


def _reporter(v: Any) -> str:
    return str(v.get("name") or v.get("username") or "") if isinstance(v, dict) else ""


def _notes(v: Any) -> str:
    if not isinstance(v, list):
        return ""
    return "\n".join(n.get("text", "") for n in v if isinstance(n, dict) and n.get("text"))


def _empty() -> AnyGeoDataFrame:
    import geopandas as gpd  # type: ignore[import-untyped]

    return cast(
        AnyGeoDataFrame, gpd.GeoDataFrame(columns=["id", "incident", "geometry"], geometry="geometry", crs="EPSG:4326")
    )


@register(tags=["io"])
def get_incident_events(
    client: EarthRangerSource,
    time_range: Annotated[TimeRange, Field(description="Incidents whose own date is in this range.")],
    max_depth: Annotated[
        int,
        AdvancedField(
            default=5,
            ge=1,
            description="How many levels of incidents inside incidents to follow (1 = direct members only).",
        ),
    ] = 5,
    include_icons: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="Embed each event type's EarthRanger icon in an `icon` column (one download per icon).",
        ),
    ] = False,
) -> Annotated[
    AnyGeoDataFrame,
    Field(description="One row per (incident, member event), with the incident's columns and an `incident` label."),
]:
    """
    Downloads the incidents (collection events) dated in the time range and every one of
    their member events, fetched by id so members dated outside the range are kept.
    Incidents inside incidents are followed; their members count towards the top-level
    incident. Adds event-type and category display names and resolves event details to
    form titles and choice labels.

    Args:
    client: The EarthRanger data source.
    time_range: Incidents whose own date is in this range.
    max_depth: How many levels of nested incidents to follow.
    include_icons: Embed event-type icons as data URIs in `icon` ("" when there is none).

    Returns:
    A GeoDataFrame with one row per (incident, member event): `incident_id`, `incident`
    ("#<serial> <title>"), `incident_serial`, `incident_title`, `incident_state`,
    `incident_time`, `incident_created_at`, `incident_updated_at`, `incident_priority`,
    then the member event's columns, `event_type_display`, `event_category_display`,
    `event_details_raw` and `icon`; sorted by incident time, then event time.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    client = as_backends(client)
    api = client.api
    incidents = _events(
        api,
        is_collection=True,
        include_related_events=True,
        since=time_range.since.isoformat(),
        until=time_range.until.isoformat(),
    )
    if incidents.empty:
        return _empty()

    links = _contains(incidents)
    root = {i: i for i in incidents["id"]}
    for inc_id, member_id in links:
        root.setdefault(member_id, root.get(inc_id, inc_id))
    members, seen = [], set(incidents["id"])
    todo = list(dict.fromkeys(m for _, m in links))
    for _ in range(max_depth):
        todo = [i for i in todo if i not in seen]
        if not todo:
            break
        seen.update(todo)
        fetched = _events_by_ids(api, todo)
        if fetched.empty:
            break
        members.append(fetched)
        new = _contains(fetched[_is_collection(fetched)])
        for inc_id, member_id in new:
            root.setdefault(member_id, root.get(inc_id, inc_id))
        links += new
        todo = [m for _, m in new]
    ev = pd.concat(members, ignore_index=True) if members else pd.DataFrame(columns=["id"])
    ev = ev[~_is_collection(ev)]  # leaves only
    returned = set(incidents["id"]).union(*(set(m["id"]) for m in members))
    # requested but not returned; links past max_depth were never requested, so aren't missing
    missing = {m for _, m in links if m in seen} - returned
    if missing:
        logger.warning("%d member events were not returned (permissions?)", len(missing))
    if ev.empty:
        return _empty()

    pairs = pd.DataFrame(
        [(root.get(i, i), m) for i, m in links], columns=["incident_id", "member_id"]
    ).drop_duplicates()
    inc = incidents.assign(
        incident="#"
        + incidents["serial_number"].astype(str)
        + " "
        + incidents["title"].fillna("(untitled)").astype(str)
    )
    inc = inc[[c for c in INCIDENT_COLUMNS if c in inc.columns]].rename(columns=INCIDENT_COLUMNS)
    out = pairs.merge(inc, on="incident_id").merge(
        ev[[c for c in MEMBER_COLUMNS if c in ev.columns]], left_on="member_id", right_on="id"
    )
    out = out.drop(columns="member_id")
    for col, fmt in (("reported_by", _reporter), ("notes", _notes)):
        out[col] = out.get(col, pd.Series(None, index=out.index)).map(fmt)
    out["time"] = pd.to_datetime(out["time"], utc=True, format="ISO8601")
    urls = out.get("image_url", pd.Series(None, index=out.index))
    out["icon"] = urls.map(_icon_data_uri) if include_icons else ""

    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    gdf = _names_and_details(client, gdf)
    return cast(AnyGeoDataFrame, gdf.sort_values(["incident_time", "time"]).reset_index(drop=True))


def _names_and_details(client: Any, gdf: Any) -> Any:
    """Event-type and category display names, and event details with form titles and choice
    labels, inside this one io task so mock-io tests replace the whole EarthRanger read."""
    try:
        gdf = client.api.get_event_type_display_names_from_events(gdf, append_category_names="never")
    except (ERClientException, requests.RequestException, KeyError):
        logger.warning("Event type display names unavailable; using event type values")
        gdf["event_type_display"] = gdf["event_type"]
    if "event_category" not in gdf.columns:
        gdf["event_category"] = None
    try:
        categories = {c["value"]: c["display"] for c in client.api.get_event_categories()}
    except (ERClientException, requests.RequestException, KeyError, TypeError):
        categories = {}
    gdf["event_category_display"] = gdf["event_category"].map(categories).fillna(gdf["event_category"])
    gdf["event_details_raw"] = gdf.get("event_details", pd.Series(None, index=gdf.index))
    return process_events_details(df=gdf, client=client, map_to_titles=True, ordered=True)
