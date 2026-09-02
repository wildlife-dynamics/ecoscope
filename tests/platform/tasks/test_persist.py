from pathlib import Path
from unittest import mock

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.platform.tasks.io import persist_df, persist_df_wrapper
from ecoscope.platform.tasks.io._persist import _hash_df


def test_persist_df_auto_filename_hashable(tmp_path):
    """Test automatic filename generation with hashable data."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")

    # Should generate a filename automatically
    dst = persist_df(df, root_path, None, "csv")

    # Verify file was created and contains correct data
    df_read = pd.read_csv(dst, index_col=0)
    pd.testing.assert_frame_equal(df_read, df)

    # Verify same dataframe generates same filename (deterministic)
    dst2 = persist_df(df, root_path, None, "csv")
    assert dst == dst2


def test_persist_df_auto_filename_unhashable(tmp_path):
    """Test automatic filename generation with unhashable data (fallback path)."""
    # Create a dataframe with unhashable types (e.g., lists)
    df = pd.DataFrame({"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]})
    root_path = str(tmp_path / "test")

    # Should generate a filename using the fallback method
    dst = persist_df(df, root_path, None, "csv")

    # Verify file was created
    df_read = pd.read_csv(dst, index_col=0)
    # Note: lists are stored as strings in CSV, so we can't directly compare
    assert len(df_read) == len(df)

    # Verify same dataframe generates same filename (deterministic)
    dst2 = persist_df(df, root_path, None, "csv")
    assert dst == dst2


def test_persist_df_auto_filename_different_data(tmp_path):
    """Test that different dataframes generate different filenames."""
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
    root_path = str(tmp_path / "test")

    dst1 = persist_df(df1, root_path, None, "csv")
    dst2 = persist_df(df2, root_path, None, "csv")

    # Different data should generate different filenames
    assert dst1 != dst2


def test_persist_df_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")
    filename = "data"
    dst = persist_df(df, root_path, filename, "csv")
    df_read = pd.read_csv(dst, index_col=0)
    pd.testing.assert_frame_equal(df_read, df)


def test_persist_df_gpkg(tmp_path):
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "geometry": [
                Point(0, 0),
                Point(1, 1),
                Point(2, 2),
            ],
        }
    )
    root_path = str(tmp_path / "test")
    filename = "data"
    dst = persist_df(df, root_path, filename, "gpkg")

    gdf_read = gpd.read_file(dst)
    pd.testing.assert_frame_equal(gdf_read, gpd.GeoDataFrame(df))


def test_persist_df_parquet(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")
    dst = persist_df(df, root_path, "data", "parquet")
    assert dst.endswith(".parquet")
    df_read = pd.read_parquet(dst)
    pd.testing.assert_frame_equal(df_read, df)


def test_persist_df_parquet_with_geometry(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"A": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
    )
    root_path = str(tmp_path / "test")
    dst = persist_df(gdf, root_path, "geo", "parquet")
    assert dst.endswith(".parquet")
    gdf_read = gpd.read_parquet(dst)
    pd.testing.assert_frame_equal(gdf_read, gdf)


def test_persist_df_json(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")
    dst = persist_df(df, root_path, "data", "json")
    assert dst.endswith(".json")
    df_read = pd.read_json(dst)
    pd.testing.assert_frame_equal(df_read, df)


def test_persist_df_geojson(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"A": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
    )
    root_path = str(tmp_path / "test")
    dst = persist_df(gdf, root_path, "geo", "geojson")
    assert dst.endswith(".geojson")
    gdf_read = gpd.read_file(dst)
    pd.testing.assert_frame_equal(gdf_read[["A", "geometry"]], gdf, check_dtype=False)


# ---------------------------------------------------------------------------
# Skip-if-exists for content-addressed filenames.
#
# The production failure these pin: within one workflow run two tasks produce
# byte-identical output, so they derive the same content hash and the second
# write overwrites the first. On a gs:// root that overwrite fails outright,
# because replacing an object needs a delete permission the workflow service
# account does not have.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filetype", ["csv", "gpkg", "parquet", "geoparquet"])
def test_persist_df_auto_filename_skips_rewrite(tmp_path, filetype):
    gdf = gpd.GeoDataFrame(
        {"A": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
    )
    root_path = str(tmp_path / "test")

    dst = persist_df(gdf, root_path, None, filetype)
    Path(dst).write_bytes(b"sentinel")

    assert persist_df(gdf, root_path, None, filetype) == dst
    assert Path(dst).read_bytes() == b"sentinel", "existing target must not be rewritten"


def test_persist_df_auto_filename_skips_serialization(tmp_path):
    """Not just the write -- the encode is skipped too."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")

    dst = persist_df(df, root_path, None, "csv")

    with mock.patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("should not serialize")):
        assert persist_df(df, root_path, None, "csv") == dst


def test_persist_df_explicit_filename_still_overwrites(tmp_path):
    """A caller-supplied name does not determine its content, so it must not skip."""
    root_path = str(tmp_path / "test")
    persist_df(pd.DataFrame({"A": [1]}), root_path, "data", "csv")
    dst = persist_df(pd.DataFrame({"A": [2]}), root_path, "data", "csv")

    assert pd.read_csv(dst, index_col=0)["A"].tolist() == [2]


@pytest.mark.parametrize(
    ("filetype", "extension"),
    [
        ("csv", ".csv"),
        ("gpkg", ".gpkg"),
        ("geoparquet", ".parquet"),
        ("parquet", ".parquet"),
        ("geojson", ".geojson"),
        ("json", ".json"),
    ],
)
def test_persist_df_extension_per_filetype(tmp_path, filetype, extension):
    gdf = gpd.GeoDataFrame({"A": [1], "geometry": [Point(0, 0)]})
    dst = persist_df(gdf, str(tmp_path / "test"), "data", filetype)

    assert dst.endswith(extension)


def test_persist_df_unsupported_filetype_raises(tmp_path):
    # Raised while resolving the extension, i.e. before any serialization.
    with pytest.raises(NotImplementedError, match="Unsupported file type"):
        persist_df(pd.DataFrame({"A": [1]}), str(tmp_path / "test"), "data", "xlsx")


def test_persist_df_hash_covers_column_labels(tmp_path):
    """Regression: `hash_pandas_object` hashes only values, never column labels.

    With column-mapping tasks upstream of persist, a rename used to yield the
    same filename for different content -- and under skip the second frame would
    silently serve the first's file.
    """
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    root_path = str(tmp_path / "test")

    assert persist_df(df, root_path, None, "csv") != persist_df(df.rename(columns={"A": "Z"}), root_path, None, "csv")


def test_persist_df_hash_covers_all_rows_for_unhashable_frames(tmp_path):
    """Regression: the fallback used to hash only shape + `head(5)`.

    Object columns holding lists/dicts (grouped event exports) take that path, so
    two frames agreeing on shape and first rows collided.
    """
    root_path = str(tmp_path / "test")
    rows = [{"k": i} for i in range(30)]
    a = pd.DataFrame({"details": rows})
    b = pd.DataFrame({"details": [*rows[:25], {"k": 999}, *rows[26:]]})

    assert a.shape == b.shape
    assert a.head(5).to_dict() == b.head(5).to_dict(), "must differ only past the old sample window"
    assert persist_df(a, root_path, None, "csv") != persist_df(b, root_path, None, "csv")


def test_persist_df_hash_distinguishes_int_from_str_in_mixed_column(tmp_path):
    """The unhashable fallback coerces with `repr`, not `str`.

    `str` would render 1 and "1" identically, collapsing two different frames
    onto one filename.
    """
    root_path = str(tmp_path / "test")
    a = pd.DataFrame({"x": [["l"], 1, "z"]})
    b = pd.DataFrame({"x": [["l"], "1", "z"]})

    assert persist_df(a, root_path, None, "csv") != persist_df(b, root_path, None, "csv")


def test_hash_df_raises_clearly_when_values_are_neither_hashable_nor_reprable():
    """The fallback's own `repr` can fail; the error must name the column.

    There is no honest name to write under in that case -- a type-derived
    placeholder collides across distinct values and an `id`-derived one changes
    per process, either of which would make skip-if-exists serve wrong bytes --
    so `_hash_df` raises rather than inventing a hash. Object-dtype hashing is
    forced to fail here so the test pins our wrapping, not pandas' internals.
    """

    class Unreprable:
        def __repr__(self):
            raise RuntimeError("boom")

    df = pd.DataFrame({"bad": [Unreprable()]})
    real = pd.util.hash_pandas_object

    def failing_on_object_dtype(obj, **kwargs):
        dtypes = obj.dtypes if isinstance(obj, pd.DataFrame) else [obj.dtype]
        if any(dtype == object for dtype in dtypes):
            raise TypeError("unhashable type")
        return real(obj, **kwargs)

    with mock.patch.object(pd.util, "hash_pandas_object", failing_on_object_dtype):
        with pytest.raises(TypeError, match="cannot content-hash column 'bad'"):
            _hash_df(df)


def test_hash_df_hot_path_avoids_the_per_column_fallback(tmp_path):
    """Ordinary frames must take the single whole-frame hash.

    The fallback feeds sha256 one array per column instead of one for the frame;
    that is affordable only because ordinary frames never reach it.
    """
    df = pd.DataFrame({"s": ["a", "b"], "i": [1, 2], "f": [1.5, 2.5]})
    real = pd.util.hash_pandas_object
    calls = []

    def counting(obj, **kwargs):
        calls.append(obj)
        return real(obj, **kwargs)

    with mock.patch.object(pd.util, "hash_pandas_object", counting):
        _hash_df(df)

    assert len(calls) == 1


def test_persist_df_wrapper_auto_filename_skips_rewrite(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3]})
    root_path = str(tmp_path / "test")

    (dst,) = persist_df_wrapper(df=df, root_path=root_path, filetypes=["csv"])
    Path(dst).write_bytes(b"sentinel")

    assert persist_df_wrapper(df=df, root_path=root_path, filetypes=["csv"]) == [dst]
    assert Path(dst).read_bytes() == b"sentinel"


def test_persist_df_wrapper_sanitize_changes_the_filename(tmp_path):
    """Regression: the hash was taken of `df` while `df_new` was written.

    `sanitize=True` and `sanitize=False` therefore shared a name for different
    bytes, so under skip one would serve the other's file.
    """
    df = pd.DataFrame({"tags": [["a", "b"], ["c"]]})
    root_path = str(tmp_path / "test")

    plain = persist_df_wrapper(df=df, root_path=root_path, filetypes=["csv"], sanitize=False)
    sanitized = persist_df_wrapper(df=df, root_path=root_path, filetypes=["csv"], sanitize=True)

    assert plain != sanitized


def test_persist_df_geoparquet_without_geometry_raises(tmp_path):
    """geopandas does not raise here -- it writes `primary_column: null` metadata
    that only a later `gpd.read_parquet` rejects. `_require_geometry` makes it loud.
    """
    with pytest.raises(ValueError, match="requires at least one geometry column"):
        persist_df(pd.DataFrame({"A": [1]}), str(tmp_path / "test"), None, "geoparquet")


@pytest.mark.parametrize("filetypes", [["parquet", "geoparquet"], ["geoparquet", "parquet"]])
def test_persist_df_wrapper_geoparquet_without_geometry_raises_either_order(tmp_path, filetypes):
    """`parquet` and `geoparquet` share `<hash>.parquet`.

    Without the guard, list order decided the outcome: parquet-first wrote a clean
    plain parquet and the geoparquet encode was skipped, geoparquet-first wrote the
    unreadable one and the plain encode was skipped -- same returned path, different
    content. The guard rejects the frame the same way whichever order it arrives in.
    """
    with pytest.raises(ValueError, match="requires at least one geometry column"):
        persist_df_wrapper(df=pd.DataFrame({"A": [1]}), root_path=str(tmp_path / "test"), filetypes=filetypes)


def test_persist_df_geoparquet_guard_runs_before_the_skip(tmp_path):
    """The guard must precede the skip probe, or an existing `<hash>.parquet` left
    by the `parquet` branch would let a geometry-less geoparquet call return it.
    """
    df = pd.DataFrame({"A": [1]})
    root_path = str(tmp_path / "test")

    plain = persist_df(df, root_path, None, "parquet")
    assert Path(plain).exists(), "the shared target must already be on disk"

    with pytest.raises(ValueError, match="requires at least one geometry column"):
        persist_df(df, root_path, None, "geoparquet")


def test_persist_df_geoparquet_with_geometry_still_writes(tmp_path):
    gdf = gpd.GeoDataFrame({"A": [1], "geometry": [Point(0, 0)]})
    dst = persist_df(gdf, str(tmp_path / "test"), None, "geoparquet")

    assert dst.endswith(".parquet")
    assert len(gpd.read_parquet(dst)) == 1


def test_persist_df_wrapper_sanitize_preserves_geodataframe(tmp_path):
    """The sanitize branch splits geometry off, sanitizes the attributes and
    reattaches -- the round trip must come back a GeoDataFrame with its CRS."""
    gdf = gpd.GeoDataFrame(
        {
            "name": ["A", "B"],
            "tags": [["x", "y"], ["z"]],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )

    dst = persist_df_wrapper(df=gdf, root_path=str(tmp_path / "test"), filetypes=["geoparquet"], sanitize=True)[0]

    written = gpd.read_parquet(dst)
    assert written.crs == gdf.crs
    assert len(written) == 2
    assert written["tags"].iloc[0] == '["x", "y"]'
    assert written["geometry"].iloc[1].equals(Point(1, 1))


def test_persist_df_wrapper_sanitize_geodataframe_with_repeat_index(tmp_path):
    """Regression: geometry was reattached with `attrs.join(df[[geom_name]])`.

    `.join` aligns on the index, so a frame carrying a non-unique index (as
    grouped/exploded frames routinely do) got a cartesian expansion and we
    persisted more rows than we were handed, with geometry paired to the
    wrong attributes.
    """
    gdf = gpd.GeoDataFrame(
        {
            "name": ["A", "B", "C"],
            "tags": [["x", "y"], ["z"], []],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=[0, 0, 1],
        crs="EPSG:4326",
    )

    dst = persist_df_wrapper(df=gdf, root_path=str(tmp_path / "test"), filetypes=["geoparquet"], sanitize=True)[0]

    written = gpd.read_parquet(dst)
    assert written.crs == gdf.crs
    assert len(written) == 3
    assert list(written["name"]) == ["A", "B", "C"]
    # Each row keeps the geometry it came in with
    assert [g.equals(e) for g, e in zip(written["geometry"], gdf["geometry"])] == [True] * 3
