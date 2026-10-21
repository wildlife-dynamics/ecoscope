import numpy as np
import pandas as pd
import pytest

from ecoscope.io.earthranger_utils import (
    decode_raw_event_details,
    normalize_column,
    patrol_segments_from_patrols_df,
    unpack_events_from_patrols_df,
)
from ecoscope.io.utils import clean_time_cols


@pytest.fixture
def df_with_times():
    return pd.DataFrame(
        data={
            "time": [
                "2023-01-30 11:26:13.805829-08:00",
                "2023-09-27T06:16:46.158966",
                "2023-09-27T06:16:46.23-07:00",
                "2023-09-27T06:16:46.1589-07:00",
                "2023-09-27T22:00:01.23-11:00",
                "2023-09-27T06:16:46.00-07:00",
                "2023-09-27T22:00:00.00-02:00",
                pd.NA,
            ]
        },
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )


def test_clean_time_cols(df_with_times):
    with pytest.raises(AttributeError):
        df_with_times["time"].dt

    cleaned = clean_time_cols(df_with_times)
    assert pd.api.types.is_datetime64_ns_dtype(cleaned["time"])
    # Check we have our dt accessor
    df_with_times["time"].dt

    expected_times = pd.arrays.DatetimeArray._from_sequence(
        [
            "2023-01-30 19:26:13.805829+00:00",
            "2023-09-27 06:16:46.158966+00:00",
            "2023-09-27 13:16:46.230000+00:00",
            "2023-09-27 13:16:46.158900+00:00",
            "2023-09-28 09:00:01.230000+00:00",
            "2023-09-27 13:16:46+00:00",
            "2023-09-28 00:00:00+00:00",
        ]
    )

    # Since the parser resolves nan's to pd.NaT,
    # and pd.NaT != pd.NaT
    # check the nan separately from the array equality
    assert np.array_equal(expected_times, cleaned["time"].array[:-1])
    assert pd.isnull(cleaned["time"]["H"])


def test_normalize_column():
    df_with_nested_column = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "details": [
                {"zebra": "z1", "apple": "a1", "middle": "m1"},
                {"zebra": "z2", "apple": "a2", "middle": "m2"},
                {"zebra": "z3", "apple": "a3", "middle": "m3"},
            ],
            "score": [10, 20, 30],
        }
    )
    # Store original column order (excluding the one to be normalized)
    original_cols = ["id", "name", "score"]

    # Normalize the 'details' column
    normalize_column(df_with_nested_column, "details")

    # Expected new columns in alphabetical order
    expected_new_cols = ["details__apple", "details__middle", "details__zebra"]

    # Verify all expected columns exist
    assert all(col in df_with_nested_column.columns for col in original_cols)
    assert all(col in df_with_nested_column.columns for col in expected_new_cols)

    # Verify values were correctly normalized
    assert list(df_with_nested_column["details__apple"]) == ["a1", "a2", "a3"]
    assert list(df_with_nested_column["details__middle"]) == ["m1", "m2", "m3"]
    assert list(df_with_nested_column["details__zebra"]) == ["z1", "z2", "z3"]

    # Verify original columns retained their values
    assert list(df_with_nested_column["id"]) == [1, 2, 3]
    assert list(df_with_nested_column["name"]) == ["Alice", "Bob", "Charlie"]
    assert list(df_with_nested_column["score"]) == [10, 20, 30]


def test_normalize_column_sorted():
    df_with_nested_column = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "details": [
                {"zebra": "z1", "apple": "a1", "middle": "m1"},
                {"zebra": "z2", "apple": "a2", "middle": "m2"},
                {"zebra": "z3", "apple": "a3", "middle": "m3"},
            ],
            "score": [10, 20, 30],
        }
    )

    # Store original column order (excluding the one to be normalized)
    original_cols = ["id", "name", "score"]

    # Normalize the 'details' column
    normalize_column(df_with_nested_column, "details", sort_columns=True)

    # Expected new columns in alphabetical order
    expected_new_cols = ["details__apple", "details__middle", "details__zebra"]

    # Verify column order: original columns first, then alphabetically sorted new columns
    expected_column_order = original_cols + expected_new_cols
    assert list(df_with_nested_column.columns) == expected_column_order


def test_decode_raw_event_details():
    df = pd.DataFrame(
        {
            "id": ["e1", "e2"],
            "event_details": [
                '{"species": "elephant", "count": 3}',
                '{"nested": {"a": 1}, "list": [1, 2]}',
            ],
        }
    )

    decode_raw_event_details(df)

    assert df["event_details"].tolist() == [
        {"species": "elephant", "count": 3},
        {"nested": {"a": 1}, "list": [1, 2]},
    ]


@pytest.mark.parametrize(
    "value",
    [None, "", pd.NA, np.nan],
    ids=["none", "empty-string", "pd-NA", "nan"],
)
def test_decode_raw_event_details_null_becomes_empty_dict(value):
    """Parity with the ER API, which serves `event_details: {}` for an event without
    details, and a precondition for pd.json_normalize (which rejects non-dicts)."""
    df = pd.DataFrame({"event_details": [value, '{"a": 1}']})

    decode_raw_event_details(df)

    assert df["event_details"].tolist() == [{}, {"a": 1}]


@pytest.mark.parametrize("value", ["not json at all", "[1, 2]", "42", '"a string"'])
def test_decode_raw_event_details_unusable_payload_becomes_empty_dict(value):
    """One malformed (or non-object) payload must not fail the whole download."""
    df = pd.DataFrame({"event_details": [value, '{"a": 1}']})

    decode_raw_event_details(df)

    assert df["event_details"].tolist() == [{}, {"a": 1}]


def test_decode_raw_event_details_is_idempotent():
    df = pd.DataFrame({"event_details": ['{"a": 1}']})

    decode_raw_event_details(df)
    decode_raw_event_details(df)

    assert df["event_details"].tolist() == [{"a": 1}]


def test_decode_raw_event_details_missing_column_is_a_noop():
    df = pd.DataFrame({"id": ["e1"]})

    decode_raw_event_details(df)

    assert list(df.columns) == ["id"]


def test_decode_raw_event_details_feeds_normalize_column():
    """The end-to-end contract the download-events workflow relies on: decoded details
    flatten into `event_details__*` columns."""
    df = pd.DataFrame(
        {
            "id": ["e1", "e2"],
            "event_details": ['{"species": "elephant"}', None],
        }
    )

    decode_raw_event_details(df)
    normalize_column(df, "event_details")

    assert df["event_details__species"].tolist() == ["elephant", np.nan]


def _patrols_df_with_events(events):
    """One patrol, one segment, carrying the given events."""
    return pd.DataFrame(
        [
            {
                "id": "p1",
                "serial_number": 1,
                "patrol_segments": [
                    {
                        "id": "seg1",
                        "patrol_type": "ecoscope_patrol",
                        "time_range": {"start_time": "2015-02-01T00:00:00+00:00"},
                        "leader": {"name": "eco_1"},
                        "events": events,
                    }
                ],
            }
        ]
    )


def _event(event_id, event_type, state, day):
    return {
        "id": event_id,
        "event_type": event_type,
        "state": state,
        "geojson": {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [36.8, -1.3]},
            "properties": {"datetime": f"2015-02-{day}T00:00:00+00:00"},
        },
    }


@pytest.fixture
def mixed_patrol_events():
    return _patrols_df_with_events(
        [
            _event("e1", "hwc_rep", "active", "01"),
            _event("e2", "hwc_rep", "resolved", "02"),
            _event("e3", "fire_rep", "active", "03"),
            _event("e4", "fire_rep", "resolved", "04"),
        ]
    )


@pytest.mark.parametrize(
    "event_state, expected",
    [
        (None, ["e1", "e2", "e3", "e4"]),
        ([], ["e1", "e2", "e3", "e4"]),
        (["active"], ["e1", "e3"]),
        (["active", "resolved"], ["e1", "e2", "e3", "e4"]),
        (["new"], []),
    ],
    ids=["none-keeps-all", "empty-keeps-all", "single", "multiple", "no-match"],
)
def test_unpack_events_from_patrols_df_event_state(mixed_patrol_events, event_state, expected):
    result = unpack_events_from_patrols_df(mixed_patrol_events, event_state=event_state)
    assert (result["id"].tolist() if len(result) else []) == expected


def test_unpack_events_from_patrols_df_event_type_and_state_are_anded(mixed_patrol_events):
    """Both filters must hold, so this keeps only the resolved hwc_rep event."""
    result = unpack_events_from_patrols_df(mixed_patrol_events, event_type=["hwc_rep"], event_state=["resolved"])
    assert result["id"].tolist() == ["e2"]


def _T(s):
    return pd.Timestamp(s).to_pydatetime()


def _leg(seg_id, start, end, members=None, is_pause=False, with_members=True):
    leg = {
        "id": seg_id,
        "patrol_type": "foot",
        "leader": {"id": "a", "name": "Asha"},
        "time_range": {"start_time": start, "end_time": end},
        "is_pause": is_pause,
    }
    if with_members:
        leg["members"] = ["a", "b", "c"] if members is None else members
    return leg


def _legs_df(legs):
    return pd.DataFrame([{"id": "p1", "serial_number": 7, "patrol_segments": legs}])


@pytest.fixture
def three_leg_df():
    return _legs_df(
        [
            _leg("seg1", "2026-01-05T08:00:00Z", "2026-01-05T12:00:00Z"),
            _leg("seg2", "2026-01-05T12:00:00Z", "2026-01-05T13:00:00Z", is_pause=True),
            _leg("seg3", "2026-01-05T13:00:00Z", "2026-01-05T16:00:00Z"),
        ]
    )


def test_patrol_segments_excludes_pauses_and_computes_durations(three_leg_df):
    legs = patrol_segments_from_patrols_df(three_leg_df)
    assert legs["patrol_segment_id"].tolist() == ["seg1", "seg3"]
    assert legs["leg_active_seconds"].tolist() == [4 * 3600.0, 3 * 3600.0]
    assert legs["leg_member_days"].tolist() == [3.0, 3.0]
    assert legs["patrol_elapsed_seconds"].tolist()[0] == 8 * 3600.0
    assert pd.isna(legs["patrol_elapsed_seconds"].iloc[1])
    assert legs["patrol_type"].iloc[0] == "foot"
    assert legs["patrol_subject"].iloc[0] == "Asha"
    assert legs["patrol_serial_number"].iloc[0] == 7


def test_patrol_segments_clips_to_window(three_leg_df):
    legs = patrol_segments_from_patrols_df(three_leg_df, since=_T("2026-01-05T10:00Z"), until=_T("2026-01-05T14:00Z"))
    assert legs["leg_active_seconds"].tolist() == [2 * 3600.0, 1 * 3600.0]
    assert legs["patrol_elapsed_seconds"].iloc[0] == 4 * 3600.0


def test_patrol_segments_drops_legs_outside_window(three_leg_df):
    legs = patrol_segments_from_patrols_df(three_leg_df, since=_T("2026-01-05T14:30Z"), until=_T("2026-01-05T15:00Z"))
    assert legs["patrol_segment_id"].tolist() == ["seg3"]
    assert (legs["leg_active_seconds"] >= 0).all()


def test_patrol_segments_open_leg_clips_end_to_until():
    open_df = _legs_df(
        [
            _leg("seg1", "2026-01-05T08:00:00Z", "2026-01-05T12:00:00Z"),
            _leg("seg3", "2026-01-05T13:00:00Z", None),
        ]
    )
    legs = patrol_segments_from_patrols_df(open_df, until=_T("2026-01-05T18:00Z"))
    assert legs["leg_active_seconds"].iloc[-1] == 5 * 3600.0
    assert patrol_segments_from_patrols_df(open_df)["patrol_segment_id"].tolist() == ["seg1"]


def test_patrol_segments_member_days_spans_midnight():
    midnight_df = _legs_df([_leg("seg1", "2026-01-05T22:00:00Z", "2026-01-06T02:00:00Z", members=["a", "b"])])
    assert patrol_segments_from_patrols_df(midnight_df)["leg_member_days"].iloc[0] == 4.0


def test_patrol_segments_without_members_gives_nan_member_days_and_no_new_columns():
    legacy_df = _legs_df([_leg("seg1", "2026-01-05T08:00:00Z", "2026-01-05T12:00:00Z", with_members=False)])
    legs = patrol_segments_from_patrols_df(legacy_df)
    assert legs["leg_member_days"].isna().all()
    assert "patrol_members" not in legs.columns and "patrol_team" not in legs.columns


def test_patrol_segments_empty_members_gives_zero_member_days():
    empty_members_df = _legs_df([_leg("seg1", "2026-01-05T08:00:00Z", "2026-01-05T12:00:00Z", members=[])])
    assert patrol_segments_from_patrols_df(empty_members_df)["leg_member_days"].iloc[0] == 0.0


def test_patrol_segments_empty_patrols_df_returns_empty():
    assert patrol_segments_from_patrols_df(pd.DataFrame()).empty


def test_patrol_segments_member_days_end_at_midnight_counts_one_day():
    df = _legs_df([_leg("seg1", "2026-01-05T22:00:00Z", "2026-01-06T00:00:00Z")])
    assert patrol_segments_from_patrols_df(df)["leg_member_days"].iloc[0] == 3.0


def test_patrol_segments_optional_columns_nan_on_legs_lacking_key():
    first = _leg("seg1", "2026-01-05T08:00:00Z", "2026-01-05T12:00:00Z")
    first["team"] = {"id": "t1"}
    first["assets"] = ["x"]
    second = _leg("seg2", "2026-01-05T13:00:00Z", "2026-01-05T16:00:00Z", with_members=False)
    legs = patrol_segments_from_patrols_df(_legs_df([first, second]))
    assert legs["patrol_team"].iloc[0] == {"id": "t1"}
    assert legs["patrol_members"].iloc[0] == ["a", "b", "c"]
    assert pd.isna(legs["patrol_team"].iloc[1]) and pd.isna(legs["patrol_members"].iloc[1])
