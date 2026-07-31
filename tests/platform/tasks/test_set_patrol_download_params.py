from ecoscope.platform.tasks.config import (
    PatrolDownloadParams,
    get_patrol_event_filename_prefix,
    get_patrol_event_filetypes,
    get_patrol_track_filename_prefix,
    get_patrol_track_filetypes,
    set_patrol_download_params,
)


def _params():
    return PatrolDownloadParams(
        track_filetypes=["parquet", "csv"],
        track_filename_prefix="tracks",
        event_filetypes=["csv"],
        event_filename_prefix="events",
    )


def test_set_patrol_download_params_returns_inputs_unchanged():
    result = set_patrol_download_params(
        track_filetypes=["parquet", "csv"],
        event_filetypes=["csv"],
        track_filename_prefix="tracks",
        event_filename_prefix="events",
    )

    assert result == _params()


def test_set_patrol_download_params_defaults():
    result = set_patrol_download_params()

    assert result.track_filetypes == ["parquet"]
    assert result.event_filetypes == ["parquet"]
    assert result.track_filename_prefix == "patrol_tracks"
    assert result.event_filename_prefix == "patrol_events"


def test_get_patrol_track_filetypes():
    params = _params()

    assert get_patrol_track_filetypes(params) is params.track_filetypes


def test_get_patrol_track_filename_prefix():
    params = _params()

    assert get_patrol_track_filename_prefix(params) is params.track_filename_prefix


def test_get_patrol_event_filetypes():
    params = _params()

    assert get_patrol_event_filetypes(params) is params.event_filetypes


def test_get_patrol_event_filename_prefix():
    params = _params()

    assert get_patrol_event_filename_prefix(params) is params.event_filename_prefix
