import os

from ecoscope.platform.serde import _gs_url_to_https_url, _persist_text


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
