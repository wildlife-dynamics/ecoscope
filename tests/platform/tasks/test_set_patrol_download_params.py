from ecoscope.platform.tasks.config import (
    get_patrol_event_filename_prefix,
    get_patrol_event_filetypes,
    get_patrol_track_filename_prefix,
    get_patrol_track_filetypes,
    set_patrol_download_params,
)


def _params():
    return (
        ["parquet", "csv"],
        "tracks",
        ["csv"],
        "events",
    )


def test_set_patrol_download_params_returns_inputs_unchanged():
    result = set_patrol_download_params(
        track_filetypes=["parquet", "csv"],
        event_filetypes=["csv"],
        track_filename_prefix="tracks",
        event_filename_prefix="events",
    )

    assert result == (["parquet", "csv"], "tracks", ["csv"], "events")


def test_set_patrol_download_params_defaults():
    result = set_patrol_download_params()

    assert result[0] == ["parquet"]  # track_filetypes
    assert result[2] == ["parquet"]  # event_filetypes
    assert result[1] == "patrol_tracks"
    assert result[3] == "patrol_events"


def test_get_patrol_track_filetypes_returns_first_element():
    params = _params()

    assert get_patrol_track_filetypes(params) is params[0]


def test_get_patrol_track_filename_prefix_returns_second_element():
    params = _params()

    assert get_patrol_track_filename_prefix(params) is params[1]


def test_get_patrol_event_filetypes_returns_third_element():
    params = _params()

    assert get_patrol_event_filetypes(params) is params[2]


def test_get_patrol_event_filename_prefix_returns_fourth_element():
    params = _params()

    assert get_patrol_event_filename_prefix(params) is params[3]
