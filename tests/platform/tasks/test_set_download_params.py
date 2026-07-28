from ecoscope.platform.tasks.config import (
    get_gps_point_filename_prefix,
    get_gps_point_filetypes,
    get_skip_relocation_persist,
    get_track_filename_prefix,
    get_track_filetypes,
    set_download_params,
)


def _params():
    return (
        ["parquet", "csv"],
        "tracks",
        ["csv"],
        "gps",
        True,
    )


def test_set_download_params_returns_inputs_unchanged():
    result = set_download_params(
        track_filetypes=["parquet", "csv"],
        download_gps_points=True,
        gps_point_filetypes=["csv"],
        track_filename_prefix="tracks",
        gps_point_filename_prefix="gps",
    )

    assert result == (["parquet", "csv"], "tracks", ["csv"], "gps", True)


def test_set_download_params_none_filetypes_default_to_parquet():
    result = set_download_params()

    assert result[0] == ["parquet"]  # track_filetypes
    assert result[2] == ["parquet"]  # gps_point_filetypes
    assert result[1] == "subject_tracks"
    assert result[3] == "relocations"
    assert result[4] is False  # download_gps_points


def test_get_track_filetypes_returns_first_element():
    params = _params()

    assert get_track_filetypes(params) is params[0]


def test_get_track_filename_prefix_returns_second_element():
    params = _params()

    assert get_track_filename_prefix(params) is params[1]


def test_get_gps_point_filetypes_returns_third_element():
    params = _params()

    assert get_gps_point_filetypes(params) is params[2]


def test_get_gps_point_filename_prefix_returns_fourth_element():
    params = _params()

    assert get_gps_point_filename_prefix(params) is params[3]


def test_get_skip_relocation_persist_negates_download_flag():
    # download_gps_points=True -> skip=False
    assert get_skip_relocation_persist(_params()) is False
    # download_gps_points=False -> skip=True
    assert get_skip_relocation_persist((["parquet"], "t", ["parquet"], "g", False)) is True
