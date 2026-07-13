"""Unit tests for the ERWarehouseClient event / patrol-event / patrols integration
in the get_events / get_patrol_events / get_patrols /
get_event_type_display_names_from_events tasks.

These use a mocked ERWarehouseClient (no live server), so unlike the integration
tests in test_io_earthranger.py they are NOT marked ``io`` and therefore run in the
default (non-io) test job -- providing coverage for the warehouse code paths on PRs.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest

from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import (
    get_event_type_display_names_from_events,
    get_events,
    get_patrol_events,
    get_patrols,
)


def _make_events_arrow_table(
    event_type_values=("hwc_rep",),
    event_category_values=("monitoring",),
    geometry_wkbs=None,
    event_times=None,
):
    """Build a pa.Table matching EVENTS_SCHEMA_V1 with one row per event type.

    ``geometry_wkbs`` optionally overrides the default point geometry (one WKB per
    event type); useful for exercising polygon -> centroid reduction.
    ``event_times`` optionally overrides the per-row event_time (tz-aware datetimes);
    useful for exercising sort-by-time parity."""
    import datetime as dt

    import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
    import pyarrow as pa
    from ecoscope_earthranger_io_core.arrow import EVENTS_SCHEMA_V1
    from shapely.geometry import Point

    n = len(event_type_values)
    event_time_values = (
        event_times if event_times is not None else [dt.datetime(2015, 6, 1, tzinfo=dt.timezone.utc)] * n
    )

    def ts_col():
        return pa.array(
            [dt.datetime(2015, 6, 1, tzinfo=dt.timezone.utc)] * n,
            type=pa.timestamp("ns", tz="UTC"),
        )

    return pa.table(
        {
            "id": [f"e{i}" for i in range(n)],
            "serial_number": pa.array(list(range(n)), type=pa.int64()),
            "event_type_id": [f"id-{v}" for v in event_type_values],
            "event_type_value": list(event_type_values),
            "event_category_value": list(event_category_values),
            "title": ["title"] * n,
            "state": ["active"] * n,
            "priority": pa.array([0] * n, type=pa.int64()),
            "event_time": pa.array(event_time_values, type=pa.timestamp("ns", tz="UTC")),
            "end_time": pa.array([None] * n, type=pa.timestamp("ns", tz="UTC")),
            "created_at": ts_col(),
            "updated_at": ts_col(),
            "is_collection": [False] * n,
            "geometry": ga.array(
                geometry_wkbs if geometry_wkbs is not None else [Point(36.8 + i, -1.3).wkb for i in range(n)]
            ),
            "reported_by": pa.array([{"id": "u1", "name": "Ranger", "type": "user"}] * n),
            "event_details": [None] * n,
            "das_tenant_id": ["tenant-a"] * n,
        },
        schema=EVENTS_SCHEMA_V1,
    )


def _make_nested_patrols_arrow_table(event_times=("2015-02-01",), event_type="hwc_rep"):
    """Build a pa.Table matching PATROLS_WITH_EVENTS_NESTED_SCHEMA_V1 (one patrol,
    one segment, one event per provided event_time)."""
    import datetime as dt

    import pyarrow as pa
    from ecoscope_earthranger_io_core.arrow import PATROLS_WITH_EVENTS_NESTED_SCHEMA_V1
    from shapely.geometry import Point

    events = [
        {
            "id": f"ev{i}",
            "serial_number": i,
            "event_type": event_type,
            "event_time": dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc),
            "priority": 0,
            "title": "title",
            "state": "active",
            "updated_at": t,
            "created_at": t,
            "geometry": Point(36.8 + i, -1.3).wkb,
            "is_collection": False,
            "event_details": None,
        }
        for i, t in enumerate(event_times)
    ]
    segment = {
        "id": "seg1",
        "patrol_type": "ecoscope_patrol",
        "patrol_type_display": "Ecoscope Patrol",
        "leader_id": "leader-1",
        "time_range_start": "2015-02-01T00:00:00+00:00",
        "time_range_end": "2015-02-01T06:00:00+00:00",
        "scheduled_start": None,
        "scheduled_end": None,
        "start_location": None,
        "end_location": None,
        "events": events,
    }
    patrol = {
        "id": "p1",
        "serial_number": 1,
        "priority": 0,
        "state": "done",
        "title": "Patrol",
        "objective": None,
        "created_at": "2015-02-01",
        "updated_at": "2015-02-01",
        "patrol_segments": [segment],
    }
    return pa.Table.from_pylist([patrol], schema=PATROLS_WITH_EVENTS_NESTED_SCHEMA_V1)


def _make_event_types_arrow_table():
    """Build a pa.Table matching EVENT_TYPES_SCHEMA_V1 with two event types that share
    a display name (so `duplicates` triggers a category append)."""
    import pyarrow as pa
    from ecoscope_earthranger_io_core.arrow import EVENT_TYPES_SCHEMA_V1

    return pa.table(
        {
            "id": ["id-1", "id-2"],
            "value": ["hwc_rep", "hwc_alt"],
            "display": ["Human Wildlife Conflict", "Human Wildlife Conflict"],
            "category_value": ["monitoring", "security"],
            "is_active": [True, True],
            "is_collection": [False, False],
        },
        schema=EVENT_TYPES_SCHEMA_V1,
    )


def _empty_events_arrow_table():
    from ecoscope_earthranger_io_core.arrow import EVENTS_SCHEMA_V1

    return EVENTS_SCHEMA_V1.empty_table()


def _assert_valid_events_gdf(df):
    """Run the EventGDF AfterValidator chain enforced at the task boundary."""
    from ecoscope.platform.schemas import _events_optional_columns, _events_strict

    return _events_strict(_events_optional_columns(df))


_EVENT_TIME_RANGE = TimeRange(
    since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
    until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
    timezone=UTC_TIMEZONEINFO,
)


def test_get_events_via_warehouse_client():
    """Warehouse branch: rename event_type_value/event_category_value/event_time,
    tz-aware time, geometry preserved, and the frame satisfies the strict schema."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            raise_on_empty=False,
        )

    mock_warehouse_client.get_events.assert_called_once()
    mock_legacy_client.get_events.assert_not_called()
    assert isinstance(result, gpd.GeoDataFrame)
    assert "event_type" in result.columns
    assert "event_category" in result.columns
    assert "time" in result.columns
    assert "event_type_value" not in result.columns
    assert result["event_type"].tolist() == ["hwc_rep"]
    assert result["event_category"].tolist() == ["monitoring"]
    assert pd.api.types.is_datetime64_ns_dtype(result["time"])
    assert result["time"].iloc[0].tzinfo is not None
    assert result["geometry"].iloc[0] is not None
    _assert_valid_events_gdf(result)


def test_get_events_via_warehouse_client_passes_slugs_no_id_round_trip():
    """The warehouse get_events takes slugs directly, so event_types are forwarded
    verbatim and no value->id lookup via get_event_types happens."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep", "fire_rep"],
            include_null_geometry=False,
            raise_on_empty=False,
        )

    call_kwargs = mock_warehouse_client.get_events.call_args.kwargs
    assert call_kwargs["event_type"] == ["hwc_rep", "fire_rep"]
    assert call_kwargs["drop_null_geometry"] is True
    mock_warehouse_client.get_event_types.assert_not_called()
    mock_legacy_client.get_event_types.assert_not_called()


def test_get_events_via_warehouse_client_synthesizes_location_from_geometry():
    """The warehouse serves no `location` column; the task synthesizes an ER-native
    location dict ({"latitude","longitude"}) from geometry so the event-details
    workflow's extract_value_from_json_column(column_name="location") keeps working."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    # single event, point at (lon=36.8, lat=-1.3)
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            # the released event-details selection includes "location"
            event_columns=["id", "time", "event_type", "event_category", "reported_by", "location", "geometry"],
            raise_on_empty=False,
        )

    location = result["location"].iloc[0]
    assert location == {"latitude": -1.3, "longitude": 36.8}
    # mirror how the event-details workflow's extract_value_from_json_column reads it
    assert location.get("latitude") == -1.3
    assert location.get("longitude") == 36.8


def test_get_events_via_warehouse_client_unavailable_column_raises_clear_error():
    """Selecting an event column the warehouse doesn't serve raises a clear error
    naming the missing column(s) and the available ones, not a bare KeyError."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        with pytest.raises(ValueError, match=r"event_columns.*'message'.*not served.*Available columns"):
            get_events(
                client=mock_legacy_client,
                time_range=_EVENT_TIME_RANGE,
                event_types=["hwc_rep"],
                event_columns=["id", "time", "message"],
                raise_on_empty=False,
            )


def test_get_events_via_warehouse_client_sorts_ascending_by_time():
    """Parity with the legacy path, which sorts events ascending by time. The
    warehouse serves event_time DESC, so the task must re-sort."""
    import datetime as dt

    later = dt.datetime(2015, 6, 3, tzinfo=dt.timezone.utc)
    earlier = dt.datetime(2015, 6, 1, tzinfo=dt.timezone.utc)

    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    # warehouse returns DESC (later first); task should flip to ascending
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table(
        event_type_values=("a", "b"),
        event_category_values=("c", "d"),
        event_times=[later, earlier],
    )

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["a", "b"],
            raise_on_empty=False,
        )

    assert result["time"].tolist() == [earlier, later]
    _assert_valid_events_gdf(result)


def test_get_events_via_warehouse_client_force_point_geometry_reduces_to_centroid():
    """Parity with the legacy path: a polygon event geometry is reduced to its
    centroid when force_point_geometry=True (the default), and preserved when False."""
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])  # centroid at (1, 1)

    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table(geometry_wkbs=[poly.wkb])

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        reduced = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            raise_on_empty=False,
        )
        # reset the mock return (from_arrow consumes it lazily; rebuild to be safe)
        mock_warehouse_client.get_events.return_value = _make_events_arrow_table(geometry_wkbs=[poly.wkb])
        preserved = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            force_point_geometry=False,
            raise_on_empty=False,
        )

    geom = reduced["geometry"].iloc[0]
    assert geom.geom_type == "Point"
    assert (geom.x, geom.y) == (1.0, 1.0)
    _assert_valid_events_gdf(reduced)

    assert preserved["geometry"].iloc[0].geom_type == "Polygon"


def test_get_events_via_warehouse_client_event_columns_subset():
    """event_columns slicing works on the renamed warehouse frame."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            event_columns=["id", "time", "event_type", "geometry"],
            raise_on_empty=False,
        )

    assert sorted(result.columns) == sorted(["id", "time", "event_type", "geometry"])
    _assert_valid_events_gdf(result)


def test_get_events_via_warehouse_client_empty():
    """Empty warehouse table: raise_on_empty raises, else returns an empty frame."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_events.return_value = _empty_events_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
            get_events(
                client=mock_legacy_client,
                time_range=_EVENT_TIME_RANGE,
                event_types=["hwc_rep"],
            )

        result = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            raise_on_empty=False,
        )
    assert result.empty
    mock_warehouse_client.get_event_types.assert_not_called()


def test_get_events_via_warehouse_client_display_values():
    """include_display_values on the warehouse path uses the local helper (not the
    stubbed client method) and populates event_type_display."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_events.return_value = _make_events_arrow_table()
    mock_warehouse_client.get_event_types.return_value = _make_event_types_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            include_display_values=True,
            raise_on_empty=False,
        )

    mock_warehouse_client.get_event_types.assert_called_once()
    mock_warehouse_client.get_event_type_display_names_from_events.assert_not_called()
    assert "event_type_display" in result.columns
    assert result["event_type_display"].tolist() == ["Human Wildlife Conflict"]


def test_get_patrols_via_warehouse_client_synthesized_shape_feeds_unpack():
    """Warehouse get_patrols is converted to the ER-native PatrolsDF shape:
    per-event synthesized geojson + per-segment time_range.start_time, and that
    shape feeds unpack_events_from_patrols_df into a valid EventGDF."""
    from ecoscope.io.earthranger_utils import unpack_events_from_patrols_df

    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrols.return_value = _make_nested_patrols_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        patrols_df = get_patrols(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            raise_on_empty=False,
        )

    mock_warehouse_client.get_patrols.assert_called_once()
    mock_legacy_client.get_patrols.assert_not_called()

    segment = patrols_df["patrol_segments"].iloc[0][0]
    assert segment["time_range"]["start_time"] == "2015-02-01T00:00:00+00:00"
    assert segment["time_range"]["end_time"] == "2015-02-01T06:00:00+00:00"
    assert segment["patrol_type"] == "ecoscope_patrol"
    # leader name is unavailable from the warehouse -> patrol_subject degrades to None
    assert segment["leader"]["name"] is None
    event = segment["events"][0]
    assert event["geojson"]["type"] == "Feature"
    assert event["geojson"]["geometry"]["type"] == "Point"
    assert event["geojson"]["properties"]["datetime"] == "2015-02-01T00:00:00+00:00"

    events = unpack_events_from_patrols_df(patrols_df, event_type=[])
    assert "event_type" in events.columns
    assert "geometry" in events.columns
    assert events["patrol_subject"].isna().all()
    _assert_valid_events_gdf(events)


def test_get_patrols_via_warehouse_client_empty():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    from ecoscope_earthranger_io_core.arrow import PATROLS_WITH_EVENTS_NESTED_SCHEMA_V1

    mock_warehouse_client.get_patrols.return_value = PATROLS_WITH_EVENTS_NESTED_SCHEMA_V1.empty_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
            get_patrols(
                client=mock_legacy_client,
                time_range=_EVENT_TIME_RANGE,
                patrol_types=["ecoscope_patrol"],
            )

        result = get_patrols(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            raise_on_empty=False,
        )
    assert result.empty


def test_get_patrol_events_via_warehouse_client():
    """Warehouse get_patrol_events derives events from get_patrols + unpack (does not
    call the warehouse get_patrol_events, honoring the no-DAG-change constraint)."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrols.return_value = _make_nested_patrols_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_patrol_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            event_types=[],
            status=None,
            raise_on_empty=False,
        )

    mock_warehouse_client.get_patrols.assert_called_once()
    mock_warehouse_client.get_patrol_events.assert_not_called()
    mock_legacy_client.get_patrol_events.assert_not_called()
    assert "event_type" in result.columns
    assert "geometry" in result.columns
    assert result["event_type"].tolist() == ["hwc_rep"]
    _assert_valid_events_gdf(result)


def test_get_patrol_events_via_warehouse_client_truncates_to_time_range():
    """truncate_to_time_range drops events outside the requested window."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    # One event in-range (2015), one out-of-range (2020).
    mock_warehouse_client.get_patrols.return_value = _make_nested_patrols_arrow_table(
        event_times=("2015-02-01", "2020-02-01")
    )

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_patrol_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            event_types=[],
            status=None,
            truncate_to_time_range=True,
            raise_on_empty=False,
        )

    assert len(result) == 1
    assert result["time"].iloc[0].year == 2015
    _assert_valid_events_gdf(result)


def test_get_patrol_events_via_warehouse_client_display_values():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrols.return_value = _make_nested_patrols_arrow_table()
    mock_warehouse_client.get_event_types.return_value = _make_event_types_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_patrol_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            event_types=[],
            status=None,
            include_display_values=True,
            raise_on_empty=False,
        )

    mock_warehouse_client.get_event_types.assert_called_once()
    mock_warehouse_client.get_event_type_display_names_from_events.assert_not_called()
    assert "event_type_display" in result.columns
    assert result["event_type_display"].tolist() == ["Human Wildlife Conflict"]


def test_get_event_type_display_names_from_events_via_warehouse_client():
    """The wrapper task resolves display names via the local helper on the warehouse
    path (the client method is a NotImplementedError stub)."""
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_event_types.return_value = _make_event_types_arrow_table()

    events_gdf = gpd.GeoDataFrame(
        {
            "event_type": ["hwc_rep"],
            "time": [pd.Timestamp("2015-06-01", tz="UTC")],
            "event_category": ["monitoring"],
            "reported_by": [{"name": "Ranger"}],
            "geometry": [None],
        }
    )

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_event_type_display_names_from_events(
            client=mock_legacy_client,
            events_gdf=events_gdf,
            append_category_names="duplicates",
        )

    mock_warehouse_client.get_event_types.assert_called_once()
    mock_legacy_client.get_event_type_display_names_from_events.assert_not_called()
    assert result["event_type_display"].tolist() == ["Human Wildlife Conflict"]


def test_get_events_warehouse_disabled_falls_back_to_legacy_client():
    """Env off: get_events uses the legacy client (value->id round-trip preserved)."""
    mock_legacy_client = MagicMock()
    mock_legacy_client.get_event_types.return_value = pd.DataFrame({"id": ["id-1"], "value": ["hwc_rep"]})
    mock_legacy_client.get_events.return_value = pd.DataFrame()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=None,
    ):
        result = get_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            event_types=["hwc_rep"],
            raise_on_empty=False,
        )

    mock_legacy_client.get_event_types.assert_called_once()
    mock_legacy_client.get_events.assert_called_once()
    assert mock_legacy_client.get_events.call_args.kwargs["event_type"] == ["id-1"]
    assert result.empty


def test_get_patrol_events_warehouse_disabled_falls_back_to_legacy_client():
    mock_legacy_client = MagicMock()
    mock_legacy_client.get_patrol_events.return_value = pd.DataFrame()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=None,
    ):
        result = get_patrol_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            event_types=[],
            status=None,
            raise_on_empty=False,
        )

    mock_legacy_client.get_patrol_events.assert_called_once()
    assert result.empty


def test_get_patrols_warehouse_disabled_falls_back_to_legacy_client():
    mock_legacy_client = MagicMock()
    mock_legacy_client.get_patrols.return_value = pd.DataFrame()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=None,
    ):
        result = get_patrols(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            status=None,
            raise_on_empty=False,
        )

    mock_legacy_client.get_patrols.assert_called_once()
    assert result.empty


def test_append_event_type_display_names_value_to_display_and_orphan_fallback():
    """value->display mapping, with orphan event types falling back to raw value."""
    from ecoscope.io.earthranger_utils import append_event_type_display_names

    events_df = pd.DataFrame({"event_type": ["hwc_rep", "orphan_type"]})
    result = append_event_type_display_names(
        events_df,
        _make_event_types_arrow_table(),
        append_category_names="never",
    )
    assert result["event_type_display"].tolist() == ["Human Wildlife Conflict", "orphan_type"]


def test_append_event_type_display_names_category_slug_substitution_on_duplicates():
    """When display names collide, the category slug (category_value) is appended as
    the category-display substitute (the warehouse serves no category display)."""
    from ecoscope.io.earthranger_utils import append_event_type_display_names

    # hwc_rep and hwc_alt share the display "Human Wildlife Conflict"
    events_df = pd.DataFrame({"event_type": ["hwc_rep", "hwc_alt"]})
    result = append_event_type_display_names(
        events_df,
        _make_event_types_arrow_table(),
        append_category_names="duplicates",
    )
    assert result["event_type_display"].tolist() == [
        "Human Wildlife Conflict (monitoring)",
        "Human Wildlife Conflict (security)",
    ]


def test_append_event_type_display_names_never_does_not_append_category():
    from ecoscope.io.earthranger_utils import append_event_type_display_names

    events_df = pd.DataFrame({"event_type": ["hwc_rep", "hwc_alt"]})
    result = append_event_type_display_names(
        events_df,
        _make_event_types_arrow_table(),
        append_category_names="never",
    )
    assert result["event_type_display"].tolist() == [
        "Human Wildlife Conflict",
        "Human Wildlife Conflict",
    ]


def test_append_event_type_display_names_orphan_category_falls_back_not_null():
    """An orphan event type (absent from the registry) has no category_value; on the
    category-append path it must fall back to the raw event_type value rather than
    letting a NaN null the whole event_type_display (which StrictEventsGDFSchema rejects)."""
    from ecoscope.io.earthranger_utils import append_event_type_display_names

    events_df = pd.DataFrame({"event_type": ["hwc_rep", "orphan_type"]})
    result = append_event_type_display_names(
        events_df,
        _make_event_types_arrow_table(),
        append_category_names="always",
    )
    assert result["event_type_display"].notna().all()
    assert result["event_type_display"].tolist() == [
        "Human Wildlife Conflict (monitoring)",
        "orphan_type (orphan_type)",
    ]


def test_synthesize_event_geojson_edge_cases():
    """_synthesize_event_geojson handles null geometry, null event_time, and a raw
    int64-nanoseconds event_time (as pyarrow to_pylist can surface nested timestamps)."""
    from shapely.geometry import Point

    from ecoscope.io.earthranger_utils import _synthesize_event_geojson

    # null geometry + null event_time
    gj = _synthesize_event_geojson({"geometry": None, "event_time": None})
    assert gj["geometry"] is None
    assert gj["properties"]["datetime"] is None

    # raw int64 ns event_time -> Timestamp path
    ts_ns = pd.Timestamp("2015-02-01T00:00:00Z").value
    gj2 = _synthesize_event_geojson({"geometry": Point(1.0, 2.0).wkb, "event_time": ts_ns})
    assert gj2["geometry"]["type"] == "Point"
    assert gj2["properties"]["datetime"].startswith("2015-02-01T00:00:00")


def test_get_patrol_events_legacy_display_values_path():
    """Warehouse disabled + include_display_values: the legacy client's
    get_event_type_display_names_from_events is used (not the warehouse helper)."""
    from shapely.geometry import Point

    legacy_events = gpd.GeoDataFrame(
        {"event_type": ["hwc_rep"], "time": [pd.Timestamp("2015-02-01T00:00:00Z")]},
        geometry=[Point(0.0, 0.0)],
        crs=4326,
    )
    mock_legacy_client = MagicMock()
    mock_legacy_client.get_patrol_events.return_value = legacy_events
    mock_legacy_client.get_event_type_display_names_from_events.return_value = legacy_events

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=None,
    ):
        get_patrol_events(
            client=mock_legacy_client,
            time_range=_EVENT_TIME_RANGE,
            patrol_types=["ecoscope_patrol"],
            event_types=["hwc_rep"],
            truncate_to_time_range=False,
            include_display_values=True,
            raise_on_empty=False,
        )

    mock_legacy_client.get_patrol_events.assert_called_once()
    mock_legacy_client.get_event_type_display_names_from_events.assert_called_once()
