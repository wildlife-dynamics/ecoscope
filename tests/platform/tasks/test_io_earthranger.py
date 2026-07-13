import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from pydantic import SecretStr
from wt_task import task

from ecoscope.platform.connections import EarthRangerClientProtocol, EarthRangerConnection
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import (
    get_analysis_field_from_event_details,
    get_analysis_field_label_from_event_details,
    get_analysis_field_unit_from_event_details,
    get_category_field_from_event_details,
    get_category_field_label_from_event_details,
    get_choices_from_v2_event_type,
    get_event_type_display_names_from_events,
    get_event_type_from_event_details,
    get_events,
    get_patrol_events,
    get_patrol_events_from_combined_params,
    get_patrol_observations,
    get_patrol_observations_from_combined_params,
    get_patrol_observations_from_patrols_df,
    get_patrol_observations_from_patrols_df_and_combined_params,
    get_patrols,
    get_patrols_from_combined_params,
    get_spatial_features_group,
    get_subjectgroup_observations,
    set_event_details_params,
    set_patrols_and_patrol_events_params,
    unpack_events_from_patrols_df,
    unpack_events_from_patrols_df_and_combined_params,
)
from ecoscope.platform.tasks.io._earthranger import (
    _EXCLUSION_FILTER_TO_INT,
    CombinedPatrolAndEventsParams,
    _make_warehouse_client_from_env,
)

pytestmark = pytest.mark.io


@pytest.fixture(scope="session")
def client():
    return EarthRangerConnection(
        server=os.environ["EARTHRANGER_SERVER"],
        username=os.environ["EARTHRANGER_USERNAME"],
        password=os.environ["EARTHRANGER_PASSWORD"],
        tcp_limit="5",
        sub_page_size="4000",
    ).get_client()


@pytest.fixture
def mock_empty_client():
    mock = MagicMock()
    mock.get_patrols.return_value = pd.DataFrame()
    mock.get_subjectgroup_observations.return_value = pd.DataFrame()
    mock.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()
    mock.get_patrol_events.return_value = pd.DataFrame()
    mock.get_events.return_value = pd.DataFrame()
    mock.get_event_types.return_value = pd.DataFrame({"id": ["AAA123"], "value": ["an_event"]})
    return mock


def test_get_subject_group_observations(client):
    result = get_subjectgroup_observations(
        client=client,
        subject_group_name="Ecoscope",
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    )

    assert len(result) > 0
    assert "geometry" in result
    assert "groupby_col" in result
    assert "fixtime" in result
    assert "junk_status" in result


def test_get_patrol_observations():
    """The task-validated entry point strips whitespace from patrol_types before the
    underlying client call. This is the wrapper behavior we want to lock in."""
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrol_observations).validate().call(
            client="MEP_DEV",
            time_range={
                "since": "2015-01-01T00:00:00Z",
                "until": "2015-03-01T23:59:59Z",
                "timezone": {
                    "label": "UTC",
                    "tzCode": "UTC",
                    "name": "UTC",
                    "utc": "+00:00",
                },
            },
            patrol_types=["    ecoscope_patrol           "],  # whitespaces are intentional to test stripping
            status=None,
            include_patrol_details=True,
            raise_on_empty=False,
        )

    call_kwargs = mock_client.get_patrol_observations_with_patrol_filter.call_args.kwargs
    assert call_kwargs["patrol_type_value"] == ["ecoscope_patrol"]


def test_get_patrol_observations_with_whitespace_in_patrol_types_without_pydantic_validations():
    """Direct (non-task) calls bypass pydantic, so whitespace is passed through as-is to
    the underlying client. This is the complement to the test above: stripping is
    pydantic's responsibility, not the wrapper's."""
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()

    get_patrol_observations(
        client=mock_client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types=["    ecoscope_patrol           "],
        status=None,
        include_patrol_details=True,
        raise_on_empty=False,
    )

    call_kwargs = mock_client.get_patrol_observations_with_patrol_filter.call_args.kwargs
    assert call_kwargs["patrol_type_value"] == ["    ecoscope_patrol           "]


def test_get_patrol_events(client):
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        event_types=[],
        status=None,
    )

    assert len(result) > 0
    assert "id" in result
    assert "event_type" in result
    assert "geometry" in result


def test_get_events(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[
            "hwc_rep",
            "bird_sighting_rep",
            "wildlife_sighting_rep",
            "poacher_camp_rep",
            "fire_rep",
            "injured_animal_rep",
        ],
        event_columns=["id", "time", "event_type", "geometry"],
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "geometry" in result


def test_get_events_with_details(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        include_details=True,
        include_updates=True,
        include_related_events=True,
        event_types=[
            "hwc_rep",
            "bird_sighting_rep",
            "wildlife_sighting_rep",
            "poacher_camp_rep",
            "fire_rep",
            "injured_animal_rep",
        ],
        event_columns=[
            "id",
            "time",
            "event_type",
            "geometry",
            "event_details",
            "updates",
            "is_linked_to",
        ],
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "geometry" in result
    assert "event_details" in result
    assert "updates" in result
    assert "is_linked_to" in result


def test_get_events_with_event_type_whitespace():
    """Task-validated event_types are stripped of whitespace before the wrapper's
    get_event_types lookup. Verify by mocking get_event_types to return clean values
    and confirming the wrapper resolved IDs (i.e., stripping happened, otherwise
    the .isin() lookup would miss everything)."""
    expected_clean = [
        "hwc_rep",
        "bird_sighting_rep",
        "wildlife_sighting_rep",
        "poacher_camp_rep",
        "fire_rep",
        "injured_animal_rep",
    ]
    event_types_df = pd.DataFrame(
        {
            "id": [f"id-{i}" for i in range(len(expected_clean))],
            "value": expected_clean,
        }
    )
    mock_client = _make_mock_client()
    mock_client.get_event_types.return_value = event_types_df
    mock_client.get_events.return_value = pd.DataFrame(
        {"id": ["e1"], "time": [pd.Timestamp("2015-06-01", tz="UTC")], "event_type": ["hwc_rep"], "geometry": [None]}
    )

    with _patched_named_connection(mock_client):
        task(get_events).validate().call(
            client="MEP_DEV",
            time_range={
                "since": "2015-01-01T00:00:00Z",
                "until": "2015-12-31T23:59:59Z",
                "timezone": {
                    "label": "UTC",
                    "tzCode": "UTC",
                    "name": "UTC",
                    "utc": "+00:00",
                },
            },
            event_types=[
                "         hwc_rep    ",  # whitespaces are intentional to test stripping
                "   bird_sighting_rep           ",
                "      wildlife_sighting_rep        ",
                "  poacher_camp_rep    ",
                "    fire_rep   ",
                "     injured_animal_rep   ",
            ],
            event_columns=["id", "time", "event_type", "geometry"],
        )

    # If stripping happened, all 6 ids should be resolved and passed to client.get_events.
    assert mock_client.get_events.call_count == 1
    resolved_ids = mock_client.get_events.call_args.kwargs["event_type"]
    assert sorted(resolved_ids) == sorted(event_types_df["id"].tolist())


def test_get_events_bad_event_type():
    """When event_types don't match any known type, the wrapper short-circuits to an
    empty DataFrame without calling get_events."""
    mock_client = _make_mock_client()
    # The known event types in this fake registry don't include "not a real type"
    mock_client.get_event_types.return_value = pd.DataFrame({"id": ["abc123"], "value": ["a_real_type"]})

    result = get_events(
        client=mock_client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=["not a real type"],
        raise_on_empty=False,
        event_columns=["id", "time", "event_type", "geometry"],
    )

    assert result.empty
    mock_client.get_events.assert_not_called()


def test_bad_token_fails():
    with pytest.raises(Exception, match="Authorization token is invalid or expired."):
        EarthRangerConnection(
            server=os.environ["EARTHRANGER_SERVER"],
            token="abc123",
            tcp_limit="5",
            sub_page_size="4000",
        ).get_client()


def test_get_patrol_observations_all_types(client):
    result = get_patrol_observations(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types=[],
        status=None,
        include_patrol_details=True,
    )
    assert "patrol_type" in result
    assert len(result["patrol_type"].unique()) > 1


def test_get_events_parity(client):
    """
    The intent of this test is to demonstrate that new args added to the signature of the get_events task
    do not alter the return value of the core library function in any way.
    """
    original_event_args = {
        "since": datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat(),
        "until": datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat(),
        "event_type_ids": ["9477c3e3-cf46-4f2e-9bdd-05f91b2201ba"],
        "drop_null_geometry": False,
    }

    # The new args added by #1058
    new_event_args = {
        "include_details": False,
        "include_updates": False,
        "include_related_events": False,
    }

    get_events_no_extras = client.get_events(**original_event_args)
    get_events_with_extras = client.get_events(**(original_event_args | new_event_args))

    pd.testing.assert_frame_equal(get_events_no_extras, get_events_with_extras)


def test_get_events_all_types(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
    )
    assert "event_type" in result
    assert len(result["event_type"].unique()) > 1


def test_get_patrol_events_all_types(client):
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types=[],
        event_types=[],
    )

    assert "patrol_type" in result
    assert len(result["patrol_type"].unique()) > 1
    assert len(result["event_type"].unique()) > 1


def test_get_patrol_events_with_event_type_filter(client):
    event_type_filter = ["hwc_rep", "fire_rep"]
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-04-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types=[],
        event_types=event_type_filter,
    )
    assert not result.empty
    assert result["event_type"].unique().tolist() == event_type_filter


@pytest.mark.parametrize(
    "filter_value, expected_int",
    list(_EXCLUSION_FILTER_TO_INT.items()),
)
def test_get_subjectgroup_observations_exclusion_filter(mock_empty_client, filter_value, expected_int):
    get_subjectgroup_observations(
        client=mock_empty_client,
        subject_group_name="Ecoscope",
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        raise_on_empty=False,
        filter=filter_value,
    )
    mock_empty_client.get_subjectgroup_observations.assert_called_once()
    call_kwargs = mock_empty_client.get_subjectgroup_observations.call_args.kwargs
    assert call_kwargs["filter"] == expected_int


def test_get_subjectgroup_observations_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "subject_group_name": "Ecoscope",
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_subjectgroup_observations(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_subjectgroup_observations(**kwargs)
    assert df.empty


def test_get_patrol_observations_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": "ecoscope_patrol",
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrol_observations(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrol_observations(**kwargs)
    assert df.empty


def test_get_patrol_events_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": "ecoscope_patrol",
        "event_types": [],
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrol_events(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrol_events(**kwargs)
    assert df.empty


def test_get_events_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "event_types": ["an_event"],
        "event_columns": ["id", "time", "event_type", "geometry"],
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_events(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_events(**kwargs)
    assert df.empty


@pytest.mark.parametrize("force_point_geometry", [True, False])
def test_get_events_forwards_force_point_geometry(mock_empty_client, force_point_geometry):
    get_events(
        client=mock_empty_client,
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=["an_event"],
        raise_on_empty=False,
        force_point_geometry=force_point_geometry,
    )
    assert mock_empty_client.get_events.call_args.kwargs["force_point_geometry"] is force_point_geometry


def test_get_events_force_point_geometry_default_is_true(mock_empty_client):
    get_events(
        client=mock_empty_client,
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=["an_event"],
        raise_on_empty=False,
    )
    assert mock_empty_client.get_events.call_args.kwargs["force_point_geometry"] is True


def test_get_patrols_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": "ecoscope_patrol",
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrols(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrols(**kwargs)
    assert df.empty


def test_get_patrol_observations_from_patrols_df_empty_response(mock_empty_client):
    empty_patrols_df = pd.DataFrame()
    kwargs = {
        "client": mock_empty_client,
        "patrols_df": empty_patrols_df,
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrol_observations_from_patrols_df(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrol_observations_from_patrols_df(**kwargs)
    assert df.empty


def test_unpack_events_from_patrols_df_empty_response(mock_empty_client):
    empty_patrols_df = pd.DataFrame()
    kwargs = {
        "patrols_df": empty_patrols_df,
        "event_types": [],
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    }

    with pytest.raises(ValueError, match=r"No event data in provided patrols_df"):
        unpack_events_from_patrols_df(**kwargs)

    kwargs["raise_on_empty"] = False
    df = unpack_events_from_patrols_df(**kwargs)
    assert df.empty


def test_patrol_events_combined():
    """The set_patrols_and_patrol_events_params -> _from_combined_params path produces
    the same underlying IO calls as the direct task path, for both patrol-observations
    and patrol-events."""
    patrol_obs_args = {
        "client": "MEP_DEV",
        "time_range": TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": ["ecoscope_patrol"],
        "status": None,
        "include_patrol_details": True,
        "raise_on_empty": False,
        "sub_page_size": 100,
    }
    patrol_events_args = {
        "client": "MEP_DEV",
        "time_range": TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": ["ecoscope_patrol"],
        "event_types": [],
        "status": None,
        "raise_on_empty": False,
        "include_null_geometry": True,
        "sub_page_size": 100,
    }
    combined_args = patrol_obs_args | patrol_events_args

    expected_patrol_obs_call_args = {
        "since": patrol_obs_args["time_range"].since.isoformat(),
        "until": patrol_obs_args["time_range"].until.isoformat(),
        "patrol_type_value": patrol_obs_args["patrol_types"],
        "status": [
            "done"  # Since status is None in the task args we expect the default value here
        ],
        "include_patrol_details": patrol_obs_args["include_patrol_details"],
        "sub_page_size": 100,
        "patrols_overlap_daterange": True,
    }
    expected_patrol_events_call_args = {
        "since": patrol_events_args["time_range"].since.isoformat(),
        "until": patrol_events_args["time_range"].until.isoformat(),
        "patrol_type_value": patrol_events_args["patrol_types"],
        "event_type": patrol_events_args["event_types"],
        "status": [
            "done"  # Since status is None in the task args we expect the default value here
        ],
        # We expect this to be inverted since this is checked against the core lib
        "drop_null_geometry": not patrol_events_args["include_null_geometry"],
        "sub_page_size": 100,
        "patrols_overlap_daterange": True,
    }

    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()
    mock_client.get_patrol_events.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrol_observations).validate().call(**patrol_obs_args)
        task(get_patrol_events).validate().call(**patrol_events_args)

        combined_params = set_patrols_and_patrol_events_params(**combined_args)
        get_patrol_events_from_combined_params(combined_params)
        get_patrol_observations_from_combined_params(combined_params)

    # Each underlying method should have been called twice (once per code path) with
    # identical args. assert_has_calls is order-preserving but allows other calls
    # in between, which is fine here.
    mock_client.get_patrol_observations_with_patrol_filter.assert_has_calls(
        [
            call(**expected_patrol_obs_call_args),
            call(**expected_patrol_obs_call_args),
        ]
    )
    mock_client.get_patrol_events.assert_has_calls(
        [
            call(**expected_patrol_events_call_args),
            call(**expected_patrol_events_call_args),
        ]
    )


# The *_combined_parity tests verify that the "task-style" entry point (e.g.
# `task(get_patrol_observations).validate().call(...)`) and the "combined-params"
# entry point (e.g. `get_patrol_observations_from_combined_params(...)`) produce
# identical underlying IO calls. That's a logical equivalence check, so it doesn't
# need a live EarthRanger server. The connection layer and warehouse selector are
# mocked so the test is fully hermetic — no login, no downloads, no flakes.


def _assert_calls_match(calls):
    """Assert that a mock was invoked exactly twice with identical args.

    Handles DataFrames in kwargs (which can't be compared with ==).
    """
    assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
    a, b = calls[0], calls[1]
    assert a.args == b.args, f"positional args differ: {a.args} vs {b.args}"
    assert a.kwargs.keys() == b.kwargs.keys(), f"kwarg keys differ: {a.kwargs.keys()} vs {b.kwargs.keys()}"
    for k in a.kwargs:
        va, vb = a.kwargs[k], b.kwargs[k]
        if isinstance(va, pd.DataFrame):
            pd.testing.assert_frame_equal(va, vb)
        else:
            assert va == vb, f"kwarg {k!r} differs: {va!r} vs {vb!r}"


def _make_mock_client():
    """Build a MagicMock that satisfies the pydantic protocol check AND can stand in
    for an EarthRangerIO instance when callers access `.server` / `.token` (the
    warehouse selector reads these before any IO method is called)."""
    mock_client = MagicMock(spec=EarthRangerClientProtocol)
    mock_client.server = "fake-server"
    mock_client.token = None
    return mock_client


def _patched_named_connection(mock_client):
    """Make `EarthRangerConnection.client_from_named_connection(<any>)` return mock_client.

    Patching at `from_named_connection` (rather than at `client_from_named_connection`)
    works around the fact that the BeforeValidator captures the bound classmethod at
    import time — but `from_named_connection` is looked up dynamically inside that
    classmethod, so patching it at runtime takes effect. As a side effect, no env
    vars are required: env reading happens inside the patched-out method.
    """
    fake_conn = MagicMock()
    fake_conn.get_client.return_value = mock_client
    return patch(
        "ecoscope.platform.connections.EarthRangerConnection.from_named_connection",
        return_value=fake_conn,
    )


def _fake_patrols_df():
    """Minimal DataFrame satisfying PatrolsDFSchema (id, state, serial_number, patrol_segments)."""
    return pd.DataFrame(
        {
            "id": ["patrol-1"],
            "state": ["done"],
            "serial_number": pd.array([1], dtype="int64"),
            "patrol_segments": [[]],
        }
    )


def test_get_patrol_observations_combined_parity():
    args = {
        "client": "MEP_DEV",
        "time_range": TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": ["ecoscope_patrol"],
        "status": None,
        "sub_page_size": 100,
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrol_observations).validate().call(**args)
        # event_types is not an arg in get_patrol_observations, but it is required for
        # CombinedPatrolAndEventsParams so explicitly add here
        combined = CombinedPatrolAndEventsParams(**args, event_types=[])
        get_patrol_observations_from_combined_params(combined)

    _assert_calls_match(mock_client.get_patrol_observations_with_patrol_filter.call_args_list)


def test_get_patrol_events_combined_parity():
    args = {
        "client": "MEP_DEV",
        "time_range": TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": ["ecoscope_patrol"],
        "event_types": [],
        "status": None,
        "sub_page_size": 100,
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrol_events.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrol_events).validate().call(**args)
        combined = CombinedPatrolAndEventsParams(**args)
        get_patrol_events_from_combined_params(combined)

    _assert_calls_match(mock_client.get_patrol_events.call_args_list)


def test_get_patrols_combined_parity():
    args = {
        "client": "MEP_DEV",
        "time_range": TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": ["ecoscope_patrol"],
        "status": None,
        "sub_page_size": 100,
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrols.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrols).validate().call(**args)
        # event_types is not an arg in get_patrols, but it is required for
        # CombinedPatrolAndEventsParams so explicitly add here
        combined = CombinedPatrolAndEventsParams(**args, event_types=[])
        get_patrols_from_combined_params(combined)

    _assert_calls_match(mock_client.get_patrols.call_args_list)


def test_get_patrol_observations_from_patrols_df_combined_parity():
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrol_types = "ecoscope_patrol"
    status = None
    patrols_df = _fake_patrols_df()

    shared_args = {
        "client": "MEP_DEV",
        "include_patrol_details": True,
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrol_observations_from_patrols_df).validate().call(**shared_args, patrols_df=patrols_df)
        combined = CombinedPatrolAndEventsParams(
            **shared_args,
            time_range=time_range,
            patrol_types=patrol_types,
            status=status,
            event_types=[],
        )
        get_patrol_observations_from_patrols_df_and_combined_params(patrols_df=patrols_df, combined_params=combined)

    _assert_calls_match(mock_client.get_patrol_observations.call_args_list)


def test_unpack_events_from_patrols_df_combined_parity():
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrols_df = _fake_patrols_df()
    shared_args = {
        "time_range": time_range,
        "event_types": [],
        "raise_on_empty": False,
    }
    mock_helper = MagicMock(return_value=pd.DataFrame())

    with patch("ecoscope.io.earthranger_utils.unpack_events_from_patrols_df", mock_helper):
        task(unpack_events_from_patrols_df).validate().call(**shared_args, patrols_df=patrols_df)
        combined = CombinedPatrolAndEventsParams(
            **shared_args,
            client="MEP_DEV",
            patrol_types="ecoscope_patrol",
            status=None,
        )
        unpack_events_from_patrols_df_and_combined_params(patrols_df=patrols_df, combined_params=combined)

    _assert_calls_match(mock_helper.call_args_list)


def test_get_patrols(client):
    result = get_patrols(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        status=None,
    )

    assert len(result) > 0
    assert "id" in result
    assert "state" in result
    assert "patrol_segments" in result
    assert "geometry" not in result


def test_get_patrol_observations_from_patrols_df(client):
    patrols = get_patrols(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        status=None,
    )
    relocs = get_patrol_observations_from_patrols_df(
        client=client,
        patrols_df=patrols,
        include_patrol_details=True,
    )

    assert len(relocs) > 0
    assert "geometry" in relocs
    assert "groupby_col" in relocs
    assert "fixtime" in relocs
    assert "junk_status" in relocs


def test_unpack_events_from_patrols_df(client):
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrols = get_patrols(
        client=client,
        time_range=time_range,
        patrol_types="ecoscope_patrol",
        status=None,
    )
    events = unpack_events_from_patrols_df(
        patrols_df=patrols,
        time_range=time_range,
        event_types=[],
    )

    assert len(events) > 0
    assert "id" in events
    assert "event_type" in events
    assert "geometry" in events


def test_get_events_empty_event_type_selection(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "geometry" in result
    assert len(result["event_type"].unique()) > 1


def test_get_event_type_display_names_from_events(client):
    events_gdf = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2025-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2025-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
    )

    with_display_types = get_event_type_display_names_from_events(
        client=client,
        events_gdf=events_gdf,
        append_category_names="duplicates",
    )
    assert "event_type_display" in with_display_types.columns


def test_get_patrol_events_with_display_values(client):
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        event_types=[],
        status=None,
        include_display_values=True,
    )

    assert len(result) > 0
    assert "id" in result
    assert "event_type" in result
    assert "event_type_display" in result
    assert "geometry" in result


def test_get_patrol_events_with_display_values_empty():
    """When the underlying client returns empty, include_display_values=True must not
    crash and must not call the display-name resolver."""
    mock_client = _make_mock_client()
    mock_client.get_patrol_events.return_value = pd.DataFrame()

    result = get_patrol_events(
        client=mock_client,
        time_range=TimeRange(
            since=datetime.strptime("1985-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("1985-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        event_types=[],
        status=None,
        raise_on_empty=False,
        include_display_values=True,
    )

    assert result.empty
    mock_client.get_event_type_display_names_from_events.assert_not_called()


def test_get_events_with_display_values(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[
            "hwc_rep",
            "bird_sighting_rep",
            "wildlife_sighting_rep",
            "poacher_camp_rep",
            "fire_rep",
            "injured_animal_rep",
        ],
        event_columns=["id", "time", "event_type", "geometry"],
        include_display_values=True,
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "event_type_display" in result
    assert "geometry" in result


def test_get_events_with_display_values_empty():
    """When get_events returns empty, include_display_values=True must not crash and
    must not call the display-name resolver."""
    mock_client = _make_mock_client()
    mock_client.get_events.return_value = pd.DataFrame()

    result = get_events(
        client=mock_client,
        time_range=TimeRange(
            since=datetime.strptime("1985-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("1985-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
        raise_on_empty=False,
        include_display_values=True,
    )

    assert result.empty
    mock_client.get_event_type_display_names_from_events.assert_not_called()


def test_get_choices_from_v2_event_type(client):
    result = get_choices_from_v2_event_type(
        client=client,
        event_type="elephant_sigthing_test",
        choice_field="herd_type",
    )

    assert {
        "bull_only": "Bull Only",
        "female_only": "Female Only",
        "mix": "Mix",
    } == result


def test_get_choices_from_v2_event_type_not_found(client):
    result = get_choices_from_v2_event_type(
        client=client,
        event_type="accident_rep",
        choice_field="herd_type",
    )

    assert {} == result


def test_get_choices_from_v2_event_type_choice_is_none(client):
    result = get_choices_from_v2_event_type(
        client=client,
        event_type="accident_rep",
        choice_field=None,
    )

    assert {} == result


def test_event_details_params_emitters(client):
    input_time_range = TimeRange(
        since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    input_event_type = "an_event_type"
    input_event_columns = ["serial_number"]
    input_analysis_field = "a_numeric_field"
    input_analysis_field_label = "Numeric Field Label"
    input_analysis_field_unit = "Fields"
    input_category_field = "an_category_field"
    input_category_field_label = "An Category Field Label"

    params = set_event_details_params(
        client=client,
        time_range=input_time_range,
        event_type=input_event_type,
        event_columns=input_event_columns,
        analysis_field=input_analysis_field,
        analysis_field_label=input_analysis_field_label,
        analysis_field_unit=input_analysis_field_unit,
        category_field=input_category_field,
        category_field_label=input_category_field_label,
    )

    assert params.get_events_params() == {
        "client": client,
        "time_range": input_time_range,
        "event_types": [input_event_type],
        "event_columns": input_event_columns,
        "include_null_geometry": True,
        "raise_on_empty": True,
        "include_details": True,
        "include_updates": False,
        "include_related_events": False,
        "include_display_values": False,
    }

    assert get_event_type_from_event_details(params) == input_event_type
    assert get_analysis_field_from_event_details(params) == input_analysis_field
    assert get_analysis_field_label_from_event_details(params) == input_analysis_field_label
    assert get_analysis_field_unit_from_event_details(params) == input_analysis_field_unit
    assert get_category_field_from_event_details(params) == input_category_field
    assert get_category_field_label_from_event_details(params) == input_category_field_label


def test_get_spatial_features_group(client):
    group_name = "mep"
    group_id = "15698426-7e0f-41df-9bc3-495d87e2e097"
    sfg = get_spatial_features_group(client=client, spatial_features_group_name=group_name)

    assert isinstance(sfg, gpd.GeoDataFrame)
    assert not sfg.empty
    assert "pk" in sfg
    assert "name" in sfg
    assert "short_name" in sfg
    assert "feature_type" in sfg
    assert "geometry" in sfg
    assert "metadata" in sfg
    assert sfg["metadata"].iloc[0]["display_name"] == group_name
    assert sfg["metadata"].iloc[0]["id"] == group_id


def test_get_fields_from_event_type_schema(client):
    fields = client.get_fields_from_event_type_schema(event_type="elephant_sigthing_test")
    assert isinstance(fields, dict)
    assert {
        "name_of_collared_elephant": "Name of collared elephant",
        "herd_type": "Herd Type",
        "how_many_identifiable_individuals": "How many identifiable individuals?",
        "how_many_elephants_are_in_the_group": "How many elephants are in the group?",
    } == fields


def test_get_fields_from_bad_event_type_schema(client):
    from ecoscope.io.earthranger import ERClientNotFound  # type: ignore[import-untyped]

    with pytest.raises(ERClientNotFound):
        client.get_fields_from_event_type_schema(event_type="  ")


# ---------------------------------------------------------------------------
# ERWarehouseClient integration tests
# ---------------------------------------------------------------------------


def _make_observations_arrow_table():
    """Build a minimal pa.Table matching OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1."""
    import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
    import pyarrow as pa
    from shapely.geometry import Point

    wkb_geom = Point(36.8, -1.3).wkb
    return pa.table(
        {
            "geometry": ga.array([wkb_geom]),
            "fixtime": pa.array(
                [pd.Timestamp("2024-01-15", tz="UTC")],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "groupby_col": pa.array(["subject-1"]),
            "extra__subject__name": pa.array(["Elephant A"]),
            "extra__subject__subject_subtype": pa.array(["elephant"]),
            "junk_status": pa.array([False]),
        },
    )


def _make_patrol_observations_arrow_table():
    """Build a minimal pa.Table matching OBSERVATIONS_WITH_PATROL_SCHEMA_SLIM_V1."""
    import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
    import pyarrow as pa
    from shapely.geometry import Point

    wkb_geom = Point(36.8, -1.3).wkb
    return pa.table(
        {
            "geometry": ga.array([wkb_geom]),
            "fixtime": pa.array(
                [pd.Timestamp("2024-01-15", tz="UTC")],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "groupby_col": pa.array(["subject-1"]),
            "extra__subject__name": pa.array(["Elephant A"]),
            "extra__subject__subject_subtype": pa.array(["elephant"]),
            "junk_status": pa.array([False]),
            "patrol_id": pa.array(["patrol-1"]),
            "patrol_title": pa.array(["Routine patrol"]),
            "patrol_serial_number": pa.array([1], type=pa.int64()),
            "patrol_status": pa.array(["done"]),
            "patrol_type__value": pa.array(["routine_patrol"]),
            "patrol_type__display": pa.array(["Routine Patrol"]),
            "patrol_start_time": pa.array(["2024-01-15T00:00:00"]),
            "patrol_end_time": pa.array(["2024-01-15T23:59:59"]),
        },
    )


_WAREHOUSE_ENV = {
    "USE_EARTHRANGER_WAREHOUSE_API": "true",
    "EARTHRANGER_WAREHOUSE_API_BASE_URL": "http://warehouse-test",
}


def test_make_warehouse_client_from_env_enabled():
    from ecoscope_earthranger_io_core.client import ERWarehouseClient

    with patch.dict(os.environ, _WAREHOUSE_ENV):
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token=SecretStr("test-token"))

    assert isinstance(result, ERWarehouseClient)
    assert result.server == "mep-dev.pamdas.org"
    assert result.warehouse_base_url == "http://warehouse-test"


def test_make_warehouse_client_from_env_missing_url():
    from ecoscope_earthranger_io_core.client import ERWarehouseClient

    env = {"USE_EARTHRANGER_WAREHOUSE_API": "true"}
    with patch.dict(os.environ, env, clear=True):
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token=SecretStr("test-token"))
    assert isinstance(result, ERWarehouseClient)
    assert result.warehouse_base_url is None


def test_make_warehouse_client_from_env_no_token():
    env = {"USE_EARTHRANGER_WAREHOUSE_API": "true"}
    with patch.dict(os.environ, env, clear=True):
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token=None)
    assert result is None


def test_get_subjectgroup_observations_via_warehouse_client():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_subjectgroup_observations.return_value = _make_observations_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_subjectgroup_observations(
            client=mock_legacy_client,
            subject_group_name="Ecoscope",
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            raise_on_empty=False,
        )

    mock_warehouse_client.get_subjectgroup_observations.assert_called_once()
    mock_legacy_client.get_subjectgroup_observations.assert_not_called()
    assert isinstance(result, gpd.GeoDataFrame)
    assert "geometry" in result
    assert "fixtime" in result
    assert "groupby_col" in result
    assert "junk_status" in result


def test_get_patrol_observations_via_warehouse_client():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrol_observations_with_patrol_filter.return_value = (
        _make_patrol_observations_arrow_table()
    )

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_patrol_observations(
            client=mock_legacy_client,
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            patrol_types=["routine_patrol"],
            raise_on_empty=False,
        )

    mock_warehouse_client.get_patrol_observations_with_patrol_filter.assert_called_once()
    mock_legacy_client.get_patrol_observations_with_patrol_filter.assert_not_called()
    call_kwargs = mock_warehouse_client.get_patrol_observations_with_patrol_filter.call_args.kwargs
    assert call_kwargs == {
        "since": "2024-01-01T00:00:00+00:00",
        "until": "2024-03-01T00:00:00+00:00",
        "patrol_type_value": ["routine_patrol"],
        "status": ["done"],
        "include_patrol_details": True,
        "sub_page_size": 100,
        "patrols_overlap_daterange": True,
    }
    assert isinstance(result, gpd.GeoDataFrame)
    assert "geometry" in result
    assert "patrol_type__value" in result


def test_get_patrol_observations_from_patrols_df_via_warehouse_client():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrol_observations.return_value = _make_patrol_observations_arrow_table()
    patrols_df = pd.DataFrame({"id": ["patrol-1"], "state": ["done"]})

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_patrol_observations_from_patrols_df(
            client=mock_legacy_client,
            patrols_df=patrols_df,
            raise_on_empty=False,
        )

    mock_warehouse_client.get_patrol_observations.assert_called_once()
    mock_legacy_client.get_patrol_observations.assert_not_called()
    call_kwargs = mock_warehouse_client.get_patrol_observations.call_args.kwargs
    assert set(call_kwargs.keys()) == {"patrols_df", "include_patrol_details", "sub_page_size"}
    assert call_kwargs["include_patrol_details"] is True
    assert call_kwargs["sub_page_size"] == 100
    assert call_kwargs["patrols_df"] is patrols_df
    assert isinstance(result, gpd.GeoDataFrame)
    assert "geometry" in result
    assert "patrol_type__value" in result


@pytest.mark.parametrize(
    "filter_value, expected_int",
    list(_EXCLUSION_FILTER_TO_INT.items()),
)
def test_get_subjectgroup_observations_via_warehouse_client_forwards_filter(filter_value, expected_int):
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_subjectgroup_observations.return_value = _make_observations_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        get_subjectgroup_observations(
            client=mock_legacy_client,
            subject_group_name="Ecoscope",
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            raise_on_empty=False,
            filter=filter_value,
        )

    mock_warehouse_client.get_subjectgroup_observations.assert_called_once()
    mock_legacy_client.get_subjectgroup_observations.assert_not_called()
    assert mock_warehouse_client.get_subjectgroup_observations.call_args.kwargs["filter"] == expected_int


@pytest.mark.parametrize("value", [True, False])
def test_get_patrol_observations_via_warehouse_client_forwards_patrols_overlap_daterange(value):
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrol_observations_with_patrol_filter.return_value = (
        _make_patrol_observations_arrow_table()
    )

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        get_patrol_observations(
            client=mock_legacy_client,
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            patrol_types=["routine_patrol"],
            raise_on_empty=False,
            patrols_overlap_daterange=value,
        )

    mock_warehouse_client.get_patrol_observations_with_patrol_filter.assert_called_once()
    mock_legacy_client.get_patrol_observations_with_patrol_filter.assert_not_called()
    assert (
        mock_warehouse_client.get_patrol_observations_with_patrol_filter.call_args.kwargs["patrols_overlap_daterange"]
        == value
    )


def test_warehouse_disabled_falls_back_to_legacy_client():
    """When warehouse is not configured, the legacy client should be used."""
    mock_legacy_client = MagicMock()
    mock_legacy_client.get_subjectgroup_observations.return_value = pd.DataFrame()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=None,
    ):
        result = get_subjectgroup_observations(
            client=mock_legacy_client,
            subject_group_name="Ecoscope",
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            raise_on_empty=False,
        )

    mock_legacy_client.get_subjectgroup_observations.assert_called_once()
    assert result.empty


# ---------------------------------------------------------------------------
# ERWarehouseClient event / patrol-event / patrols integration tests
# ---------------------------------------------------------------------------


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
