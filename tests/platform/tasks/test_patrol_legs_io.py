"""Mock-only tests for the patrol leg IO tasks. Unmarked so they run in CI
(`-m "not io and not smart_io"`); the live mep-dev test stays in test_io_earthranger.py.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from wt_task import task

from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import (
    add_patrol_roster_names,
    add_patrol_team_names,
    add_patrol_type_display_names,
    get_patrol_segments_from_patrols_df,
    get_patrol_segments_from_patrols_df_and_combined_params,
)
from ecoscope.platform.tasks.io._earthranger import CombinedPatrolAndEventsParams

UTC_TZ_DICT = {"label": "UTC", "tzCode": "UTC", "name": "UTC", "utc": "+00:00"}


def _three_leg_patrols_df():
    def leg(seg_id, start, end, is_pause=False):
        return {
            "id": seg_id,
            "patrol_type": "foot",
            "leader": {"id": "a", "name": "Asha"},
            "time_range": {"start_time": start, "end_time": end},
            "is_pause": is_pause,
            "members": ["a", "b", "c"],
        }

    return pd.DataFrame(
        [
            {
                "id": "p1",
                "state": "done",
                "serial_number": 7,
                "patrol_segments": [
                    leg("seg1", "2026-01-05T08:00:00Z", "2026-01-05T12:00:00Z"),
                    leg("seg2", "2026-01-05T12:00:00Z", "2026-01-05T13:00:00Z", is_pause=True),
                    leg("seg3", "2026-01-05T13:00:00Z", "2026-01-05T16:00:00Z"),
                ],
            }
        ]
    )


def test_get_patrol_segments_from_patrols_df_task_validates_and_truncates():
    result = (
        task(get_patrol_segments_from_patrols_df)
        .validate()
        .call(
            patrols_df=_three_leg_patrols_df(),
            time_range={"since": "2026-01-05T10:00:00Z", "until": "2026-01-05T14:00:00Z", "timezone": UTC_TZ_DICT},
        )
    )
    assert result["leg_active_seconds"].tolist() == [2 * 3600.0, 1 * 3600.0]


def test_get_patrol_segments_from_patrols_df_no_truncation():
    result = (
        task(get_patrol_segments_from_patrols_df)
        .validate()
        .call(
            patrols_df=_three_leg_patrols_df(),
            time_range={"since": "2026-01-05T10:00:00Z", "until": "2026-01-05T14:00:00Z", "timezone": UTC_TZ_DICT},
            truncate_to_time_range=False,
        )
    )
    assert result["leg_active_seconds"].tolist() == [4 * 3600.0, 3 * 3600.0]


def test_get_patrol_segments_from_patrols_df_open_leg_ends_at_until_even_without_truncation():
    patrols_df = _three_leg_patrols_df()
    patrols_df["patrol_segments"].iloc[0][2]["time_range"]["end_time"] = None
    result = (
        task(get_patrol_segments_from_patrols_df)
        .validate()
        .call(
            patrols_df=patrols_df,
            time_range={"since": "2026-01-05T10:00:00Z", "until": "2026-01-05T14:00:00Z", "timezone": UTC_TZ_DICT},
            truncate_to_time_range=False,
        )
    )
    assert result["patrol_segment_id"].tolist() == ["seg1", "seg3"]
    assert result["leg_active_seconds"].tolist() == [4 * 3600.0, 1 * 3600.0]


def test_get_patrol_segments_from_patrols_df_member_days_use_time_range_timezone():
    patrols_df = _three_leg_patrols_df()
    patrols_df["patrol_segments"].iloc[0][0]["time_range"] = {
        "start_time": "2026-01-05T19:00:00Z",
        "end_time": "2026-01-05T23:00:00Z",
    }
    nairobi = {"label": "Nairobi", "tzCode": "Africa/Nairobi", "name": "EAT", "utc": "+03:00"}
    result = (
        task(get_patrol_segments_from_patrols_df)
        .validate()
        .call(
            patrols_df=patrols_df,
            time_range={"since": "2026-01-05T00:00:00Z", "until": "2026-01-06T00:00:00Z", "timezone": nairobi},
        )
    )
    assert result["leg_member_days"].tolist()[0] == 6.0  # 3 members x 2 local days


def test_get_patrol_segments_from_patrols_df_raise_on_empty():
    tr = TimeRange(
        since=datetime(2026, 1, 5, tzinfo=timezone.utc),
        until=datetime(2026, 1, 6, tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    with pytest.raises(ValueError, match="No patrol segments"):
        get_patrol_segments_from_patrols_df(patrols_df=pd.DataFrame(), time_range=tr, raise_on_empty=True)


def test_get_patrol_segments_from_patrols_df_combined_params():
    combined = CombinedPatrolAndEventsParams(
        time_range=TimeRange(
            since=datetime(2026, 1, 5, 10, tzinfo=timezone.utc),
            until=datetime(2026, 1, 5, 14, tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        client="MEP_DEV",
        patrol_types="ecoscope_patrol",
        status=None,
        event_types=[],
    )
    result = get_patrol_segments_from_patrols_df_and_combined_params(
        patrols_df=_three_leg_patrols_df(), combined_params=combined
    )
    assert result["leg_active_seconds"].tolist() == [2 * 3600.0, 1 * 3600.0]


@pytest.fixture
def roster_client():
    mock = MagicMock()
    mock.get_patrol_teams.return_value = pd.DataFrame({"id": ["t1", "t2"], "name": ["Alpha", "Bravo"]})
    names = {f"m{i}": f"Name {i}" for i in range(60)}
    names.update({"s1": "Asha", "s2": "Ben", "v1": "Land Cruiser"})

    def _subjects(id, **kwargs):
        found = [i for i in id.split(",") if i in names]
        return pd.DataFrame({"id": found, "name": [names[i] for i in found]})

    mock.get_subjects.side_effect = _subjects
    return mock


def test_add_patrol_team_names_resolves_and_falls_back(roster_client):
    df = pd.DataFrame({"patrol_team": ["t1", "t9", None]})
    out = add_patrol_team_names(df, roster_client)
    assert out["patrol_team_name"].tolist() == ["Alpha", "t9", "None"]


def test_add_patrol_team_names_without_column_fills_none(roster_client):
    out = add_patrol_team_names(pd.DataFrame({"x": [1, 2]}), roster_client)
    assert out["patrol_team_name"].tolist() == ["None", "None"]
    roster_client.get_patrol_teams.assert_not_called()


def test_add_patrol_roster_names_resolves_lists(roster_client):
    df = pd.DataFrame({"patrol_members": [["s1", "s2"], []], "patrol_assets": [["v1"], ["zz"]]})
    out = add_patrol_roster_names(df, roster_client)
    assert out["patrol_member_names"].tolist() == [["Asha", "Ben"], []]
    assert out["patrol_asset_names"].tolist() == [["Land Cruiser"], ["zz"]]
    roster_client.get_subjects.assert_called_once()
    kwargs = roster_client.get_subjects.call_args.kwargs
    assert kwargs["include_inactive"] is True
    assert set(kwargs["id"].split(",")) == {"s1", "s2", "v1", "zz"}


def test_add_patrol_roster_names_resolves_more_than_fifty_ids(roster_client):
    ids = [f"m{i}" for i in range(60)]
    df = pd.DataFrame({"patrol_members": [ids], "patrol_assets": [[]]})
    out = add_patrol_roster_names(df, roster_client)
    assert out["patrol_member_names"].iloc[0] == [f"Name {i}" for i in range(60)]
    roster_client.get_subjects.assert_called_once()


def test_add_patrol_roster_names_without_columns_fills_none(roster_client):
    out = add_patrol_roster_names(pd.DataFrame({"x": [1]}), roster_client)
    assert out["patrol_member_names"].tolist() == ["None"] and out["patrol_asset_names"].tolist() == ["None"]
    roster_client.get_subjects.assert_not_called()


def test_add_patrol_roster_names_nothing_resolves_falls_back_to_ids(roster_client):
    roster_client.get_subjects.side_effect = None
    roster_client.get_subjects.return_value = pd.DataFrame()
    df = pd.DataFrame({"patrol_members": [["m1", "m2"]], "patrol_assets": [[]]})
    out = add_patrol_roster_names(df, roster_client)
    assert out["patrol_member_names"].tolist() == [["m1", "m2"]]


def test_add_patrol_roster_names_accepts_arrays(roster_client):
    df = pd.DataFrame({"patrol_members": [np.array(["s1"])], "patrol_assets": [np.array([], dtype=str)]})
    out = add_patrol_roster_names(df, roster_client)
    assert out["patrol_member_names"].tolist() == [["Asha"]]
    assert out["patrol_asset_names"].tolist() == [[]]


def test_add_patrol_team_names_unnamed_team_falls_back_to_id(roster_client):
    roster_client.get_patrol_teams.return_value = pd.DataFrame({"id": ["t1", "t3"], "name": ["Alpha", None]})
    out = add_patrol_team_names(pd.DataFrame({"patrol_team": ["t1", "t3"]}), roster_client)
    assert out["patrol_team_name"].tolist() == ["Alpha", "t3"]


@pytest.fixture
def patrol_types_client():
    mock = MagicMock()
    mock.get_patrol_types.return_value = pd.DataFrame(
        {"value": ["foot", "car"], "display": ["Foot Patrol", "Car Patrol"]},
        index=pd.Index(["pt1", "pt2"], name="id"),
    )
    return mock


def test_add_patrol_type_display_names_maps_value_to_display(patrol_types_client):
    df = pd.DataFrame({"patrol_type": ["foot", "car", "foot"]})
    out = add_patrol_type_display_names(df, patrol_types_client)
    assert out["patrol_type"].tolist() == ["Foot Patrol", "Car Patrol", "Foot Patrol"]
    assert out["patrol_type__value"].tolist() == ["foot", "car", "foot"]
    patrol_types_client.get_patrol_types.assert_called_once_with(include_inactive=True)
    assert df["patrol_type"].tolist() == ["foot", "car", "foot"]


def test_add_patrol_type_display_names_unmatched_falls_back_to_value(patrol_types_client):
    df = pd.DataFrame({"patrol_type": ["boat", "foot"]})
    out = add_patrol_type_display_names(df, patrol_types_client)
    assert out["patrol_type"].tolist() == ["boat", "Foot Patrol"]


def test_add_patrol_type_display_names_keeps_existing_value_column(patrol_types_client):
    df = pd.DataFrame({"patrol_type": ["foot"], "patrol_type__value": ["kept"]})
    out = add_patrol_type_display_names(df, patrol_types_client)
    assert out["patrol_type__value"].tolist() == ["kept"]
    assert out["patrol_type"].tolist() == ["Foot Patrol"]


def test_add_patrol_type_display_names_without_column_is_noop(patrol_types_client):
    df = pd.DataFrame({"x": [1]})
    out = add_patrol_type_display_names(df, patrol_types_client)
    assert out.equals(df)
    patrol_types_client.get_patrol_types.assert_not_called()


def test_add_patrol_type_display_names_example_return_loads():
    from wt_task.testing import create_func_magicmock

    out = create_func_magicmock("ecoscope.platform.tasks.io", "add_patrol_type_display_names")()
    assert (out["patrol_type"] == "Ecoscope Leg Test").all()
    assert "patrol_type__value" in out.columns
