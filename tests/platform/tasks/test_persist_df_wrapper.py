import json
import os

import numpy as np
import pandas as pd

from ecoscope.platform.tasks.io._persist_df_wrapper import (
    _hash_grouper_key,
    persist_grouped_dfs_for_results_download,
)
from ecoscope.platform.tasks.transformation._sanitize import (
    _decode_bytes,
    _has_bytes,
    _has_collections,
    _has_numbers,
    _has_strings,
    _isnull,
    _jsonify,
    _looks_numeric_series,
    sanitize_for_arrow,
)


class TestIsNull:
    """Tests for _isnull helper function."""

    def test_isnull_none(self):
        """Test that None is recognized as null."""
        assert _isnull(None) is True

    def test_isnull_nan(self):
        """Test that NaN is recognized as null."""
        assert _isnull(float("nan")) is True

    def test_isnull_pandas_na(self):
        """Test that pandas NA is recognized as null."""
        assert _isnull(pd.NA) is True

    def test_isnull_valid_values(self):
        """Test that valid values are not null."""
        assert _isnull(0) is False
        assert _isnull("") is False
        # Note: _isnull is designed for scalar values, not collections
        # Lists/arrays return numpy arrays from pd.isna, so we test scalars only
        assert _isnull(False) is False
        assert _isnull(1) is False
        assert _isnull("text") is False

    def test_isnull_dataframe_not_null(self):
        """Test that DataFrame values don't cause ambiguous truth value errors."""
        # This is a regression test for the "ambiguous truth value" error
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Should return False (DataFrames are treated as non-null objects)
        assert _isnull(df) is False

    def test_isnull_series_not_null(self):
        """Test that Series values don't cause ambiguous truth value errors."""
        series = pd.Series([1, 2, 3])
        # Should return False (Series are treated as non-null objects)
        assert _isnull(series) is False

    def test_isnull_numpy_types(self):
        """Test that numpy types are handled correctly."""
        assert _isnull(np.int64(42)) is False
        assert _isnull(np.float64(3.14)) is False
        assert _isnull(np.nan) is True
        assert _isnull(np.bool_(True)) is False


class TestDecodeBytes:
    """Tests for _decode_bytes helper function."""

    def test_decode_bytes_utf8(self):
        """Test decoding UTF-8 bytes."""
        result = _decode_bytes(b"hello world")
        assert result == "hello world"

    def test_decode_bytes_with_unicode(self):
        """Test decoding bytes with unicode characters."""
        result = _decode_bytes("hello 世界".encode("utf-8"))
        assert result == "hello 世界"

    def test_decode_bytearray(self):
        """Test decoding bytearray."""
        result = _decode_bytes(bytearray(b"test"))
        assert result == "test"

    def test_decode_non_bytes(self):
        """Test that non-bytes values are returned unchanged."""
        assert _decode_bytes("string") == "string"
        assert _decode_bytes(123) == 123
        assert _decode_bytes([1, 2, 3]) == [1, 2, 3]


class TestJsonify:
    """Tests for _jsonify helper function."""

    def test_jsonify_list(self):
        """Test converting list to JSON string."""
        result = _jsonify([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_jsonify_dict(self):
        """Test converting dict to JSON string."""
        result = _jsonify({"key": "value"})
        assert result == '{"key": "value"}'

    def test_jsonify_set(self):
        """Test converting set to JSON string."""
        # Note: The actual _jsonify implementation may convert set to list internally
        # or handle sets differently. Let's test what it actually does.
        result = _jsonify({1, 2, 3})
        # json.dumps on a set directly will fail, so the implementation
        # likely needs to handle it. For now, verify it's a string.
        assert isinstance(result, str)
        # Parse and verify contents
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert set(parsed) == {1, 2, 3}

    def test_jsonify_non_collection(self):
        """Test that non-collection values are returned unchanged."""
        assert _jsonify("string") == "string"
        assert _jsonify(123) == 123
        assert _jsonify(None) is None

    def test_jsonify_numpy_array(self):
        """Test converting numpy array to JSON string."""
        arr = np.array([1, 2, 3])
        result = _jsonify(arr)
        assert result == "[1, 2, 3]"

    def test_jsonify_nested_numpy_in_list(self):
        """Test converting list with nested numpy arrays to JSON string."""
        data = [1, np.array([2, 3]), 4]
        result = _jsonify(data)
        parsed = json.loads(result)
        assert parsed == [1, [2, 3], 4]

    def test_jsonify_nested_numpy_in_dict(self):
        """Test converting dict with numpy arrays to JSON string."""
        data = {"key": np.array([1, 2, 3])}
        result = _jsonify(data)
        parsed = json.loads(result)
        assert parsed == {"key": [1, 2, 3]}

    def test_jsonify_numpy_scalar_in_list(self):
        """Test converting list with numpy scalars to JSON string."""
        data = [np.int64(1), np.float64(2.5), np.bool_(True)]
        result = _jsonify(data)
        parsed = json.loads(result)
        assert parsed == [1, 2.5, True]


class TestLooksNumericSeries:
    """Tests for _looks_numeric_series helper function."""

    def test_looks_numeric_integers(self):
        """Test series with only integers."""
        s = pd.Series([1, 2, 3, 4])
        assert _looks_numeric_series(s)

    def test_looks_numeric_floats(self):
        """Test series with only floats."""
        s = pd.Series([1.1, 2.2, 3.3])
        assert _looks_numeric_series(s)

    def test_looks_numeric_mixed_numbers(self):
        """Test series with mixed integers and floats."""
        s = pd.Series([1, 2.5, 3, 4.7])
        assert _looks_numeric_series(s)

    def test_looks_numeric_with_na(self):
        """Test series with numeric values and NaN."""
        s = pd.Series([1, 2, np.nan, 4])
        assert _looks_numeric_series(s)

    def test_looks_numeric_strings(self):
        """Test series with strings."""
        s = pd.Series(["1", "2", "3"])
        assert not _looks_numeric_series(s)

    def test_looks_numeric_empty(self):
        """Test empty series."""
        s = pd.Series([])
        assert not _looks_numeric_series(s)

    def test_looks_numeric_all_na(self):
        """Test series with all NaN values."""
        s = pd.Series([np.nan, np.nan, np.nan])
        assert not _looks_numeric_series(s)


class TestHasBytes:
    """Tests for _has_bytes helper function."""

    def test_has_bytes_true(self):
        """Test series containing bytes."""
        s = pd.Series([b"hello", "world"])
        assert _has_bytes(s)

    def test_has_bytes_bytearray(self):
        """Test series containing bytearray."""
        s = pd.Series([bytearray(b"test"), "other"])
        assert _has_bytes(s)

    def test_has_bytes_false(self):
        """Test series without bytes."""
        s = pd.Series(["string1", "string2"])
        assert not _has_bytes(s)

    def test_has_bytes_with_na(self):
        """Test series with bytes and NaN."""
        s = pd.Series([b"hello", np.nan, "world"])
        assert _has_bytes(s)


class TestHasCollections:
    """Tests for _has_collections helper function."""

    def test_has_collections_list(self):
        """Test series containing lists."""
        s = pd.Series([[1, 2], "other"])
        assert _has_collections(s)

    def test_has_collections_dict(self):
        """Test series containing dicts."""
        s = pd.Series([{"key": "value"}, "other"])
        assert _has_collections(s)

    def test_has_collections_set(self):
        """Test series containing sets."""
        s = pd.Series([{1, 2, 3}, "other"])
        assert _has_collections(s)

    def test_has_collections_false(self):
        """Test series without collections."""
        s = pd.Series([1, 2, "string"])
        assert not _has_collections(s)


class TestHasStrings:
    """Tests for _has_strings helper function."""

    def test_has_strings_true(self):
        """Test series containing strings."""
        s = pd.Series(["hello", 123])
        assert _has_strings(s)

    def test_has_strings_false(self):
        """Test series without strings."""
        s = pd.Series([1, 2, 3])
        assert not _has_strings(s)

    def test_has_strings_with_na(self):
        """Test series with strings and NaN."""
        s = pd.Series(["hello", np.nan, 123])
        assert _has_strings(s)


class TestHasNumbers:
    """Tests for _has_numbers helper function."""

    def test_has_numbers_integers(self):
        """Test series containing integers."""
        s = pd.Series([1, 2, "string"])
        assert _has_numbers(s)

    def test_has_numbers_floats(self):
        """Test series containing floats."""
        s = pd.Series([1.5, 2.7, "string"])
        assert _has_numbers(s)

    def test_has_numbers_false(self):
        """Test series without numbers."""
        s = pd.Series(["hello", "world"])
        assert not _has_numbers(s)


class TestSanitizeForArrow:
    """Tests for sanitize_for_arrow function."""

    def test_sanitize_numeric_columns(self):
        """Test that numeric columns are preserved."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
            }
        )
        result = sanitize_for_arrow(df)
        assert pd.api.types.is_integer_dtype(result["int_col"])
        assert pd.api.types.is_float_dtype(result["float_col"])

    def test_sanitize_datetime_columns(self):
        """Test that datetime columns are preserved."""
        df = pd.DataFrame(
            {
                "date_col": pd.date_range("2023-01-01", periods=3),
            }
        )
        result = sanitize_for_arrow(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date_col"])

    def test_sanitize_boolean_columns(self):
        """Test that boolean columns are preserved."""
        df = pd.DataFrame(
            {
                "bool_col": [True, False, True],
            }
        )
        result = sanitize_for_arrow(df)
        assert pd.api.types.is_bool_dtype(result["bool_col"])

    def test_sanitize_bytes_to_string(self):
        """Test that bytes are converted to strings."""
        df = pd.DataFrame(
            {
                "bytes_col": [b"hello", b"world", None],
            }
        )
        result = sanitize_for_arrow(df)
        assert result["bytes_col"].dtype == "string[pyarrow]"
        assert result["bytes_col"][0] == "hello"

    def test_sanitize_collections_to_json(self):
        """Test that collections are converted to JSON strings."""
        df = pd.DataFrame(
            {
                "list_col": [[1, 2], [3, 4], None],
            }
        )
        result = sanitize_for_arrow(df)
        assert result["list_col"].dtype == "string[pyarrow]"
        assert result["list_col"][0] == "[1, 2]"

    def test_sanitize_mixed_numeric_string(self):
        """Test that mixed numeric and string columns become strings."""
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", 3],
            }
        )
        result = sanitize_for_arrow(df)
        assert result["mixed_col"].dtype == "string[pyarrow]"
        assert result["mixed_col"][1] == "two"

    def test_sanitize_force_string_cols(self):
        """Test force_string_cols parameter."""
        # force_string_cols only applies to object dtype columns, not numeric dtypes
        # So we need to test with an object column
        df = pd.DataFrame(
            {
                "col1": pd.Series([1, 2, 3], dtype=object),
                "col2": [b"bytes", b"data", None],
            }
        )
        result = sanitize_for_arrow(df, force_string_cols={"col1"})
        assert result["col1"].dtype == "string[pyarrow]"
        assert result["col1"][0] == "1"

    def test_sanitize_numeric_with_nulls(self):
        """Test numeric columns with null values."""
        # When we have object dtype with numeric values and nulls
        df = pd.DataFrame(
            {
                "col": pd.Series([1, None, 3], dtype=object),
            }
        )
        result = sanitize_for_arrow(df)
        # Should convert to nullable Int64
        assert result["col"].dtype == "Int64"

    def test_sanitize_object_to_numeric(self):
        """Test object dtype that's actually numeric gets converted."""
        df = pd.DataFrame(
            {
                "col": pd.Series([1, 2, 3], dtype=object),
            }
        )
        result = sanitize_for_arrow(df)
        # Should be converted to numeric type
        assert pd.api.types.is_numeric_dtype(result["col"])

    def test_sanitize_categorical(self):
        """Test that categorical columns are preserved."""
        df = pd.DataFrame(
            {
                "cat_col": pd.Categorical(["a", "b", "c"]),
            }
        )
        result = sanitize_for_arrow(df)
        assert isinstance(result["cat_col"].dtype, pd.CategoricalDtype)

    def test_sanitize_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        result = sanitize_for_arrow(df)
        assert len(result) == 0

    def test_sanitize_dict_to_json(self):
        """Test that dictionaries are converted to JSON strings."""
        df = pd.DataFrame(
            {
                "dict_col": [{"a": 1}, {"b": 2}, None],
            }
        )
        result = sanitize_for_arrow(df)
        assert result["dict_col"].dtype == "string[pyarrow]"
        assert result["dict_col"][0] == '{"a": 1}'

    def test_sanitize_numpy_arrays_in_list(self):
        """Test that lists containing numpy arrays are converted to JSON strings."""
        df = pd.DataFrame(
            {
                "array_col": [[1, 2], np.array([3, 4]), None],
            }
        )
        result = sanitize_for_arrow(df)
        assert result["array_col"].dtype == "string[pyarrow]"
        assert result["array_col"][0] == "[1, 2]"
        # Parse the JSON to verify the numpy array was properly converted
        parsed = json.loads(result["array_col"][1])
        assert parsed == [3, 4]

    def test_sanitize_dict_with_numpy_values(self):
        """Test that dicts containing numpy arrays/scalars are converted to JSON strings."""
        df = pd.DataFrame(
            {
                "dict_col": [
                    {"arr": np.array([1, 2, 3])},
                    {"val": np.int64(42)},
                    None,
                ],
            }
        )
        result = sanitize_for_arrow(df)
        assert result["dict_col"].dtype == "string[pyarrow]"
        # Verify the numpy array was properly converted
        parsed0 = json.loads(result["dict_col"][0])
        assert parsed0 == {"arr": [1, 2, 3]}
        # Verify the numpy scalar was properly converted
        parsed1 = json.loads(result["dict_col"][1])
        assert parsed1 == {"val": 42}


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
