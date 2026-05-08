import hashlib
import os

from ecoscope.platform.tasks.io import persist_text, persist_text_v2


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


def test_persist_text_v2_explicit_filename(tmp_path):
    text = '{"layers": []}'
    root_path = str(tmp_path / "test")
    filename = "map.json"
    dst = persist_text_v2(text, root_path, filename=filename)
    with open(dst) as f:
        assert f.read() == text
    assert dst == os.path.join(root_path, filename)


def test_persist_text_v2_default_extension_is_html(tmp_path):
    text = "<div>map</div>"
    root_path = str(tmp_path / "test")
    dst = persist_text_v2(text, root_path)
    with open(dst) as f:
        assert f.read() == text
    expected_filename = hashlib.sha256(text.encode()).hexdigest()[:7] + ".html"
    assert dst == os.path.join(root_path, expected_filename)


def test_persist_text_v2_json_extension(tmp_path):
    text = '{"layers": []}'
    root_path = str(tmp_path / "test")
    dst = persist_text_v2(text, root_path, extension="json")
    with open(dst) as f:
        assert f.read() == text
    expected_filename = hashlib.sha256(text.encode()).hexdigest()[:7] + ".json"
    assert dst == os.path.join(root_path, expected_filename)


def test_persist_text_v2_extension_ignored_when_filename_given(tmp_path):
    text = '{"layers": []}'
    root_path = str(tmp_path / "test")
    dst = persist_text_v2(text, root_path, extension="json", filename="map.geojson")
    assert dst == os.path.join(root_path, "map.geojson")


def test_persist_text_v2_generated_filename_with_suffix_uses_extension(tmp_path):
    text = '{"layers": []}'
    root_path = str(tmp_path / "test")
    dst = persist_text_v2(text, root_path, extension="json", filename_suffix="grouped")
    expected_filename = hashlib.sha256(text.encode()).hexdigest()[:7] + "_grouped.json"
    assert dst == os.path.join(root_path, expected_filename)
