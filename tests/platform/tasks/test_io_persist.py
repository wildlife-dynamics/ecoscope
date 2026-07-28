import hashlib
import json
import os

import pandas as pd

from ecoscope.platform.tasks.io import persist_json, persist_text
from ecoscope.platform.tasks.io._persist import (
    _hash_grouper_key,
    persist_grouped_dfs_for_results_download,
)


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


def test_persist_json_explicit_filename(tmp_path):
    data = {"layers": [], "viewState": {"zoom": 1}}
    root_path = str(tmp_path / "test")
    dst = persist_json(data, root_path, filename="map.json")
    with open(dst) as f:
        assert json.load(f) == data
    assert dst == os.path.join(root_path, "map.json")


def test_persist_json_generated_filename(tmp_path):
    data = {"layers": [], "viewState": {"zoom": 1}}
    root_path = str(tmp_path / "test")
    dst = persist_json(data, root_path)
    with open(dst) as f:
        assert json.load(f) == data
    expected_filename = hashlib.sha256(json.dumps(data).encode()).hexdigest()[:7] + ".json"
    assert dst == os.path.join(root_path, expected_filename)


def test_persist_json_appends_extension_when_filename_has_none(tmp_path):
    data = {"layers": []}
    root_path = str(tmp_path / "test")
    dst = persist_json(data, root_path, filename="map")
    assert dst == os.path.join(root_path, "map.json")


def test_persist_json_with_suffix(tmp_path):
    data = {"layers": []}
    root_path = str(tmp_path / "test")
    dst = persist_json(data, root_path, filename="map.json", filename_suffix="v2")
    assert dst == os.path.join(root_path, "map_v2.json")


def test_persist_json_generated_filename_with_suffix(tmp_path):
    data = {"layers": []}
    root_path = str(tmp_path / "test")
    dst = persist_json(data, root_path, filename_suffix="grouped")
    expected_filename = hashlib.sha256(json.dumps(data).encode()).hexdigest()[:7] + "_grouped.json"
    assert dst == os.path.join(root_path, expected_filename)


def test_persist_json_accepts_basemodel(tmp_path):
    from ecoscope.platform.tasks.results._pydeck import DeckJsonSpec

    spec = DeckJsonSpec(
        layers=[],
        initialViewState={"longitude": 0, "latitude": 0, "zoom": 1},
        views={"@@type": "MapView"},
    )
    root_path = str(tmp_path / "test")
    dst = persist_json(spec, root_path, filename="map.json")
    with open(dst) as f:
        loaded = json.load(f)
    assert loaded["layers"] == []
    assert loaded["initialViewState"] == {"longitude": 0, "latitude": 0, "zoom": 1}
    assert loaded["views"] == {"@@type": "MapView"}


class TestPersistDfForResultsDownload:
    """Tests for persist_grouped_dfs_for_results_download."""

    def test_filename_layout_prefix_keyhash_dfhash(self, tmp_path):
        # Layout: <filename_prefix>_<key_hash>_<df_hash>.<ext>
        df = pd.DataFrame({"a": [1, 2]})
        key = (("event_type", "=", "wildlife_sighting"),)
        expected_key_hash = _hash_grouper_key(key)

        paths = persist_grouped_dfs_for_results_download(
            grouped_dfs=[(key, df)],
            root_path=str(tmp_path),
            filetypes=["parquet"],
            filename_prefix="events",
        )
        assert len(paths) == 1
        name = os.path.basename(paths[0])
        stem, ext = os.path.splitext(name)
        assert ext == ".parquet"
        parts = stem.split("_")
        # ["events", "<key_hash>", "<df_hash>"]
        assert parts[0] == "events"
        assert parts[1] == expected_key_hash
        assert len(parts[2]) == 7
        assert os.path.exists(paths[0])

    def test_no_prefix_uses_keyhash_then_dfhash(self, tmp_path):
        key = (("event_type", "=", "carcass"),)
        expected_key_hash = _hash_grouper_key(key)
        paths = persist_grouped_dfs_for_results_download(
            grouped_dfs=[(key, pd.DataFrame({"a": [1]}))],
            root_path=str(tmp_path),
            filetypes=["parquet"],
        )
        stem, _ = os.path.splitext(os.path.basename(paths[0]))
        parts = stem.split("_")
        # ["<key_hash>", "<df_hash>"]
        assert parts[0] == expected_key_hash
        assert len(parts) == 2

    def test_each_group_written_to_distinct_file(self, tmp_path):
        iterables = [
            ((("event_type", "=", "wildlife_sighting"),), pd.DataFrame({"a": [1, 2]})),
            ((("event_type", "=", "carcass"),), pd.DataFrame({"a": [3, 4]})),
        ]
        paths = persist_grouped_dfs_for_results_download(
            grouped_dfs=iterables,
            root_path=str(tmp_path),
            filetypes=["parquet"],
            filename_prefix="events",
        )
        assert len(paths) == 2
        assert len(set(paths)) == 2
        for p in paths:
            assert os.path.exists(p)

    def test_multiple_filetypes_per_group(self, tmp_path):
        iterables = [((("event_type", "=", "carcass"),), pd.DataFrame({"a": [1]}))]
        paths = persist_grouped_dfs_for_results_download(
            grouped_dfs=iterables,
            root_path=str(tmp_path),
            filetypes=["csv", "parquet"],
            filename_prefix="events",
        )
        assert len(paths) == 2
        suffixes = {os.path.splitext(p)[1] for p in paths}
        assert suffixes == {".csv", ".parquet"}

    def test_empty_iterable_returns_empty_list(self, tmp_path):
        assert persist_grouped_dfs_for_results_download(grouped_dfs=[], root_path=str(tmp_path)) == []
