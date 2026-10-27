import os

import pytest

from ecoscope.platform import serde
from ecoscope.platform.serde import (
    _get_path,
    _gs_url_to_https_url,
    _my_content_type,
    _persist_bytes,
    _persist_text,
    _read_path_if_file_exists,
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


def test_read_path_if_file_exists_none_when_absent(tmp_path) -> None:
    assert _read_path_if_file_exists(str(tmp_path / "root"), "nope.txt") is None


def test_read_path_if_file_exists_matches_persist_text_return(tmp_path) -> None:
    root_path = str(tmp_path / "root")
    dst = _persist_text("hi", root_path, "name.txt")

    assert _read_path_if_file_exists(root_path, "name.txt") == dst


def test_read_path_if_file_exists_none_when_target_is_dir(tmp_path) -> None:
    # Pins `is_file()` over `exists()`. Pairs with
    # `test_persist_text_failure_when_target_is_dir`: a directory must not read as
    # an existing file, so a content-addressed write still raises rather than
    # silently skipping.
    root_path = str(tmp_path / "root")
    os.makedirs(os.path.join(root_path, "name.txt"))

    assert _read_path_if_file_exists(root_path, "name.txt") is None


def test_read_path_if_file_exists_none_for_zero_byte_file(tmp_path) -> None:
    # A zero-length local file is the usual artifact of a crash mid-write, since
    # local writes truncate before they write. Treat it as absent so the next run
    # repairs it. GCS uploads are atomic, so the guard is local-only.
    root_path = str(tmp_path / "root")
    _persist_text("", root_path, "empty.txt")

    assert _read_path_if_file_exists(root_path, "empty.txt") is None


def test_read_path_if_file_exists_creates_missing_root_dir(tmp_path) -> None:
    # Documents an inherited side effect of `_get_path`, not a desired one.
    root_path = tmp_path / "created-by-probe"
    assert _read_path_if_file_exists(str(root_path), "x.txt") is None
    assert root_path.is_dir()


def test_read_path_if_file_exists_unsupported_scheme_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported scheme"):
        _read_path_if_file_exists("s3://bucket/path", "x.txt")


def test_read_path_if_file_exists_is_scheme_agnostic(monkeypatch) -> None:
    """Cover the non-local branch without needing GCS.

    The repo has no GS mocking anywhere, so this pins our logic (probe the write
    path, never stat a non-local one) rather than cloudpathlib's.
    """

    class FakeCloudPath:
        def is_file(self):
            return True

    read_path = "https://storage.googleapis.com/bucket/dir/x.txt"
    monkeypatch.setattr(serde, "_get_path", lambda root, name: (FakeCloudPath(), read_path))

    # Not a local `Path`, so the zero-byte guard must not try to stat it.
    assert _read_path_if_file_exists("gs://bucket/dir", "x.txt") == read_path


def test_persist_text_always_overwrites(tmp_path) -> None:
    # `_persist_*` must stay unconditional overwrites: skipping is a task-layer
    # policy that only applies to content-addressed names.
    root_path = str(tmp_path / "root")
    _persist_text("first", root_path, "name.txt")
    dst = _persist_text("second", root_path, "name.txt")

    with open(dst) as f:
        assert f.read() == "second"


def test_persist_bytes_always_overwrites(tmp_path) -> None:
    root_path = str(tmp_path / "root")
    _persist_bytes(b"first", root_path, "name.bin")
    dst = _persist_bytes(b"second", root_path, "name.bin")

    with open(dst, "rb") as f:
        assert f.read() == b"second"
