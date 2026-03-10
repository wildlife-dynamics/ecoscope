import hashlib
import os

from ecoscope.platform.tasks.io import persist_text


def test_persist_text(tmp_path):
    text = "<div>map</div>"
    root_path = str(tmp_path / "test")
    filename = "map.html"
    dst = persist_text(text, root_path, filename)
    with open(dst) as f:
        assert f.read() == text
    assert dst == os.path.join(root_path, filename)


def test_persist_text_generated_filename(tmp_path):
    text = "<div>map</div>"
    root_path = str(tmp_path / "test")
    dst = persist_text(text, root_path)
    with open(dst) as f:
        assert f.read() == text
    expected_filename = hashlib.sha256(text.encode()).hexdigest()[:7] + ".html"
    assert dst == os.path.join(root_path, expected_filename)


def test_persist_text_with_suffix(tmp_path):
    text = "<div>map</div>"
    root_path = str(tmp_path / "test")
    filename = "map.html"
    suffix = "v2"
    expected_filename = "map_v2.html"
    dst = persist_text(text, root_path, filename, suffix)
    with open(dst) as f:
        assert f.read() == text
    assert dst == os.path.join(root_path, expected_filename)


def test_persist_text_generated_filename_with_suffix(tmp_path):
    text = "<div>map</div>"
    root_path = str(tmp_path / "test")
    suffix = "v2"
    dst = persist_text(text, root_path, filename_suffix=suffix)
    with open(dst) as f:
        assert f.read() == text
    expected_filename = hashlib.sha256(text.encode()).hexdigest()[:7] + "_v2.html"
    assert dst == os.path.join(
        root_path,
        expected_filename,
    )
