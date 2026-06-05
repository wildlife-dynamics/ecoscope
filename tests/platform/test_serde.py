import os

import pytest

from ecoscope.platform.serde import (
    _get_path,
    _gs_url_to_https_url,
    _my_content_type,
    _persist_bytes,
    _persist_text,
)


def test_persist_text(tmp_path):
    text = "<div>map</div>"
    root_path = str(tmp_path / "test")
    filename = "map.html"
    dst = _persist_text(text, root_path, filename)
    with open(dst) as f:
        assert f.read() == text
    assert dst == os.path.join(root_path, filename)


def test_gs_url_to_https_url():
    gs_url = "gs://bucket/path/to/file"
    https_url = "https://storage.googleapis.com/bucket/path/to/file"
    assert _gs_url_to_https_url(gs_url) == https_url


def test_my_content_type_html() -> None:
    assert _my_content_type("foo.html") == ("text/html", None)


def test_persist_bytes_round_trip(tmp_path) -> None:
    data = b"binary-payload"
    root_path = str(tmp_path / "out")
    dst = _persist_bytes(data, root_path, "x.bin")

    with open(dst, "rb") as f:
        assert f.read() == data
    assert dst == os.path.join(root_path, "x.bin")


def test_persist_text_failure_when_target_is_dir(tmp_path) -> None:
    root_path = str(tmp_path / "rooted")
    os.makedirs(os.path.join(root_path, "name.txt"))

    with pytest.raises(ValueError, match="Failed to write text"):
        _persist_text("hi", root_path, "name.txt")


def test_persist_bytes_failure_when_target_is_dir(tmp_path) -> None:
    root_path = str(tmp_path / "rooted")
    os.makedirs(os.path.join(root_path, "name.bin"))

    with pytest.raises(ValueError, match="Failed to write bytes"):
        _persist_bytes(b"data", root_path, "name.bin")


def test_get_path_unsupported_scheme_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported scheme"):
        _get_path("s3://bucket/path", "x.txt")
