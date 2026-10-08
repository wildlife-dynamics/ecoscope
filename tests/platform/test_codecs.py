"""Conformance suite for the `wt_task.codecs` entry point.

The headline property is `hit == miss`: whatever a codec stores must come back
indistinguishable from what went in, and the same value must digest the same
however it was encoded. Everything here is one of those two halves.
"""

import json
import warnings

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pytest
from shapely.geometry import Point
from wt_task.cache import get_registry
from wt_task.skip import SKIP_SENTINEL

from ecoscope.platform.codecs import (
    ARROW_CODEC_ID,
    FRAME_CODEC_ID,
    GUARD_CODEC_ID,
    KEYED_CODEC_ID,
    UNDIGESTIBLE,
    digest_dataframe,
)
from ecoscope.platform.indexes import SpatialGrouper


@pytest.fixture
def registry():
    """The process-wide registry, rebuilt so plugin discovery is exercised."""
    get_registry.cache_clear()
    try:
        yield get_registry()
    finally:
        get_registry.cache_clear()


def roundtrip(registry, value):
    codec = registry.for_value(value)
    assert codec is not None, f"no codec claims {type(value).__name__}"
    return codec.decode(codec.encode(value))


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def test_entry_point_is_discovered_without_warning():
    """A plugin that fails to load is disabled with a warning, not an error.

    Which means a broken codec module and a correctly-registered one look
    identical from the outside -- every round-trip test below would still pass
    while nothing was cached. Assert discovery positively instead.
    """
    get_registry.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry = get_registry()
        assert [w for w in caught if "codec" in str(w.message).lower()] == []
        assert registry.for_id(FRAME_CODEC_ID) is not None
        assert registry.for_value(pd.DataFrame({"a": [1]})).codec_id == FRAME_CODEC_ID
    finally:
        get_registry.cache_clear()


def test_geodataframe_resolves_to_the_frame_codec_by_mro(registry):
    """Claiming `pd.DataFrame` covers its subclasses; no priority needed."""
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    assert registry.for_value(gdf).codec_id == FRAME_CODEC_ID


# --------------------------------------------------------------------------- #
# frame round-trips -- one per fidelity case
# --------------------------------------------------------------------------- #


def test_roundtrip_preserves_a_non_default_crs(registry):
    gdf = gpd.GeoDataFrame({"a": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:3857")
    out = roundtrip(registry, gdf)
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == gdf.crs
    assert out.equals(gdf)


def test_roundtrip_preserves_a_multiindex(registry):
    """`merge_df` sets the index to composite filter keys, which `index=False` drops."""
    index = pd.MultiIndex.from_tuples([("a", "1"), ("b", "2")], names=["grouper", "value"])
    df = pd.DataFrame({"x": [1, 2]}, index=index)
    out = roundtrip(registry, df)
    assert list(out.index.names) == ["grouper", "value"]
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_preserves_tz_aware_timestamps(registry):
    df = pd.DataFrame({"t": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])})
    out = roundtrip(registry, df)
    assert out["t"].dtype == df["t"].dtype
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_preserves_dict_cells(registry):
    """ER event details arrive as dicts in an object column."""
    df = pd.DataFrame({"details": [{"k": 1}, {"k": 2, "j": "x"}]})
    out = roundtrip(registry, df)
    assert isinstance(out["details"][0], dict)
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_preserves_tuple_cells(registry):
    """RGBA colours are tuples, and parquet hands them back as `ndarray`.

    The inverse hazard is the reason decode reads a recorded spec rather than
    guessing: restoring every sequence as a tuple would corrupt genuine lists.
    """
    df = pd.DataFrame({"rgba": [(1, 2, 3, 255), (4, 5, 6, 255)]})
    out = roundtrip(registry, df)
    assert out["rgba"].tolist() == [(1, 2, 3, 255), (4, 5, 6, 255)]
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_keeps_lists_and_tuples_apart_in_the_same_frame(registry):
    df = pd.DataFrame({"t": [(1, 2), (3, 4)], "l": [[1, 2], [3, 4]]})
    out = roundtrip(registry, df)
    assert out["t"].tolist() == [(1, 2), (3, 4)]
    assert out["l"].tolist() == [[1, 2], [3, 4]]


def test_roundtrip_handles_a_column_mixing_lists_and_tuples(registry):
    df = pd.DataFrame({"mixed": [(1, 2), [3, 4], None]})
    out = roundtrip(registry, df)
    assert out["mixed"][0] == (1, 2)
    assert out["mixed"][1] == [3, 4]
    assert out["mixed"][2] is None


def test_roundtrip_of_an_empty_geodataframe(registry):
    """`EmptyDataFrame` is a real return type, not just an edge case."""
    gdf = gpd.GeoDataFrame(
        {"a": pd.Series([], dtype="int64")},
        geometry=gpd.GeoSeries([], dtype="geometry"),
        crs="EPSG:4326",
    )
    out = roundtrip(registry, gdf)
    assert len(out) == 0
    assert out.crs == gdf.crs


def test_roundtrip_of_a_frame_carrying_attrs(registry):
    df = pd.DataFrame({"a": [1]})
    df.attrs = {"source": "warehouse"}
    out = roundtrip(registry, df)
    assert out.attrs == {"source": "warehouse"}


def test_roundtrip_of_a_frame_without_geometry_returns_a_plain_dataframe(registry):
    df = pd.DataFrame({"a": [1, 2]})
    out = roundtrip(registry, df)
    assert type(out) is pd.DataFrame
    pd.testing.assert_frame_equal(out, df)


# --------------------------------------------------------------------------- #
# frame digest -- stability and sensitivity
# --------------------------------------------------------------------------- #


def test_digest_is_stable_while_the_encoded_bytes_need_not_be(registry):
    """The whole reason `digest` is a separate callable from `encode`.

    Parquet is not byte-stable in general; a digest derived from it would
    invalidate every downstream step on every run.
    """
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    codec = registry.for_value(df)
    assert codec.digest(df) == codec.digest(df.copy())
    assert codec.digest(df) == digest_dataframe(df)


def test_digest_distinguishes_frames_differing_only_in_crs():
    """The latent bug in `_persist_df`'s skip-if-exists, made a test.

    Two frames identical but for their projection used to hash equal, so the
    skip served a file in the wrong CRS.
    """
    points = [Point(0, 0), Point(1, 1)]
    a = gpd.GeoDataFrame({"x": [1, 2]}, geometry=points, crs="EPSG:4326")
    b = gpd.GeoDataFrame({"x": [1, 2]}, geometry=points, crs="EPSG:3857")
    assert digest_dataframe(a) != digest_dataframe(b)


def test_digest_distinguishes_frames_differing_only_in_attrs():
    a = pd.DataFrame({"x": [1]})
    b = pd.DataFrame({"x": [1]})
    b.attrs = {"units": "metres"}
    assert digest_dataframe(a) != digest_dataframe(b)


def test_digest_distinguishes_frames_differing_only_in_index():
    a = pd.DataFrame({"x": [1, 2]}, index=["p", "q"])
    b = pd.DataFrame({"x": [1, 2]}, index=["r", "s"])
    assert digest_dataframe(a) != digest_dataframe(b)


def test_digest_distinguishes_frames_differing_only_in_index_names():
    a = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["p", "q"], name="site"))
    b = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["p", "q"], name="region"))
    assert digest_dataframe(a) != digest_dataframe(b)


def test_digest_of_a_frame_holding_a_resolved_spatial_grouper(registry):
    """`SpatialGrouper` compares equal resolved or not, so never lean on that.

    It is not itself a cached value, but it reaches the cache inside frames and
    `df.attrs`, where only its content can be allowed to speak.
    """
    regions = gpd.GeoDataFrame({"r": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    resolved = SpatialGrouper(spatial_index_name="regions")
    resolved.resolve(regions)
    unresolved = SpatialGrouper(spatial_index_name="regions")

    # The hazard itself, pinned: equality and hashing see only the name.
    assert resolved == unresolved

    a = pd.DataFrame({"x": [1]})
    a.attrs = {"grouper": resolved.index_name, "resolved": digest_dataframe(regions)}
    b = pd.DataFrame({"x": [1]})
    b.attrs = {"grouper": unresolved.index_name, "resolved": None}
    assert digest_dataframe(a) != digest_dataframe(b)


def test_digest_refuses_rather_than_raising_on_an_unhashable_value(registry):
    """`digest` runs outside wt-task's try/except, so it must never raise.

    `digest_dataframe` does raise on cells that are neither hashable nor
    reprable -- correct for a filename, fatal for a cache -- so the codec
    returns the refusal marker and `encode` turns it into "not cached".
    """

    class Unreprable:
        def __repr__(self):
            raise RuntimeError("boom")

    df = pd.DataFrame({"bad": [Unreprable()]})
    codec = registry.for_value(df)
    assert codec.digest(df) == UNDIGESTIBLE
    with pytest.raises(TypeError):
        codec.encode(df)


# --------------------------------------------------------------------------- #
# keyed iterables
# --------------------------------------------------------------------------- #


def test_keyed_iterable_roundtrips_keys_as_tuples(registry):
    """JSON would hand the keys back as lists, which `split_groups` consumers aren't."""
    df = pd.DataFrame({"a": [1, 2]})
    value = [
        ((("month", "=", "January"),), df),
        ((("month", "=", "February"),), df.iloc[:1]),
    ]
    out = roundtrip(registry, value)
    assert registry.for_value(value).codec_id == KEYED_CODEC_ID
    assert [k for k, _ in out] == [k for k, _ in value]
    assert all(isinstance(k, tuple) and isinstance(k[0], tuple) for k, _ in out)
    pd.testing.assert_frame_equal(out[0][1], df)


def test_keyed_iterable_roundtrips_combined_values(registry):
    """`groupbykey` nests a list of values under each key."""
    df = pd.DataFrame({"a": [1]})
    value = [((("x", "=", "1"),), [df, "a path", 3])]
    out = roundtrip(registry, value)
    pd.testing.assert_frame_equal(out[0][1][0], df)
    assert out[0][1][1:] == ["a path", 3]


def test_keyed_iterable_roundtrips_a_skip_sentinel(registry):
    """`groupbykey_passthrough_skip` keeps the key and collapses values to a sentinel."""
    value = [((("x", "=", "1"),), SKIP_SENTINEL)]
    out = roundtrip(registry, value)
    assert out[0][1] is SKIP_SENTINEL


def test_keyed_iterable_roundtrips_a_null_key(registry):
    df = pd.DataFrame({"a": [1]})
    out = roundtrip(registry, [(None, df)])
    assert out[0][0] is None


def test_keyed_iterable_digest_is_stable_and_sensitive(registry):
    df = pd.DataFrame({"a": [1, 2]})
    codec = registry.for_value([((("x", "=", "1"),), df)])
    same = codec.digest([((("x", "=", "1"),), df.copy())])
    assert codec.digest([((("x", "=", "1"),), df)]) == same
    assert codec.digest([((("x", "=", "2"),), df)]) != same
    assert codec.digest([((("x", "=", "1"),), df.iloc[:1])]) != same


def test_keyed_iterable_refuses_values_no_codec_claims(registry):
    """Claimed on key shape, so an unencodable value is refused, never stringified."""
    value = [((("x", "=", "1"),), object())]
    codec = registry.for_value(value)
    assert codec.codec_id == KEYED_CODEC_ID
    assert codec.digest(value) == UNDIGESTIBLE
    with pytest.raises(TypeError):
        codec.encode(value)


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        ["a", "b"],
        [],
        [{"a": 1}],
    ],
)
def test_ordinary_lists_still_fall_through_to_json(registry, value):
    """A declining `matches` is a fall-through, not a refusal."""
    assert registry.for_value(value).codec_id == "json/v1"


# --------------------------------------------------------------------------- #
# the JSON guard
# --------------------------------------------------------------------------- #


def test_a_list_json_cannot_represent_is_refused_rather_than_stringified(registry):
    """`set_groupers` returns `list[ValueGrouper | TemporalGrouper | SpatialGrouper]`.

    `json/v1` encodes with `default=str`, so without the guard a hit would hand
    `split_groups` a list of reprs instead of groupers.
    """
    groupers = [SpatialGrouper(spatial_index_name="regions")]
    codec = registry.for_value(groupers)
    assert codec.codec_id == GUARD_CODEC_ID
    assert codec.digest(groupers) == UNDIGESTIBLE
    with pytest.raises(TypeError):
        codec.encode(groupers)


def test_the_guard_leaves_json_native_values_to_json(registry):
    assert registry.for_value({"a": [1, "b", None]}).codec_id == "json/v1"
    assert registry.for_value([{"path": "/tmp/x.csv"}]).codec_id == "json/v1"


def test_the_guard_claims_a_list_holding_a_tuple(registry):
    """JSON has no tuple, so `json/v1` would return a list and break `hit == miss`."""
    assert registry.for_value([("a", 1)]).codec_id == GUARD_CODEC_ID


# --------------------------------------------------------------------------- #
# arrow tables
# --------------------------------------------------------------------------- #


def arrow_table():
    """A table carrying both role and geoarrow extension field metadata."""
    schema = pa.schema(
        [
            pa.field("count", pa.int64(), metadata={b"role": b"measure"}),
            pa.field("geometry", pa.binary(), metadata={b"ARROW:extension:name": b"geoarrow.wkb"}),
        ],
        metadata={b"ecoscope": b"v1"},
    )
    return pa.table([pa.array([1, 2]), pa.array([b"a", b"b"])], schema=schema)


def test_arrow_table_roundtrips_through_ipc_with_metadata_intact(registry):
    table = arrow_table()
    assert registry.for_value(table).codec_id == ARROW_CODEC_ID
    out = roundtrip(registry, table)
    assert out.equals(table)
    assert out.schema.field("count").metadata == {b"role": b"measure"}
    assert out.schema.field("geometry").metadata == {b"ARROW:extension:name": b"geoarrow.wkb"}
    assert out.schema.metadata == {b"ecoscope": b"v1"}


def test_arrow_digest_is_stable_across_encodings_and_chunking(registry):
    """IPC bytes move with chunk layout; identity must not."""
    table = arrow_table()
    codec = registry.for_value(table)
    chunked = pa.concat_tables([table.slice(0, 1), table.slice(1, 1)])
    assert codec.digest(table) == codec.digest(arrow_table())
    assert codec.digest(chunked) == codec.digest(table)
    assert codec.encode(chunked) != codec.encode(table)


def test_arrow_digest_is_sensitive_to_values_and_to_field_metadata(registry):
    table = arrow_table()
    codec = registry.for_value(table)
    other_values = pa.table([pa.array([1, 3]), pa.array([b"a", b"b"])], schema=table.schema)
    stripped = table.replace_schema_metadata(None)
    assert codec.digest(other_values) != codec.digest(table)
    assert codec.digest(stripped) != codec.digest(table)


def test_the_two_frame_codecs_do_not_collide(registry):
    """A table and an equivalent frame each dispatch to their own codec."""
    df = pd.DataFrame({"a": [1, 2]})
    table = pa.Table.from_pandas(df, preserve_index=False)
    assert registry.for_value(df).codec_id == FRAME_CODEC_ID
    assert registry.for_value(table).codec_id == ARROW_CODEC_ID
    # Namespaced by codec id, so the same logical data does not dedup across
    # formats -- the price of not defining a canonical form across two type
    # systems.
    assert registry.for_value(df).digest(df) != registry.for_value(table).digest(table)


# --------------------------------------------------------------------------- #
# the shape of the payload
# --------------------------------------------------------------------------- #


def test_frame_payload_is_readable_parquet(registry):
    """Kept a valid (geo)parquet file so existing readers can consume cache objects."""
    import pyarrow.parquet as pq

    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    payload = registry.for_value(gdf).encode(gdf)
    assert payload[:4] == b"PAR1"
    assert b"geo" in (pq.read_schema(pa.BufferReader(payload)).metadata or {})


def test_keyed_payload_embeds_frames_raw_rather_than_base64(registry):
    df = pd.DataFrame({"a": list(range(100))})
    codec = registry.for_value([((("x", "=", "1"),), df)])
    payload = codec.encode([((("x", "=", "1"),), df)])
    frame_payload = registry.for_value(df).encode(df)
    assert frame_payload in payload
    header = payload[payload.index(b"{") : payload.index(b"PAR1")]
    assert json.loads(header)["k"] == "list"


def test_roundtrip_preserves_dicts_with_different_keys_per_row(registry):
    """Arrow types a dict column as a struct over the *union* of its keys.

    So an ER event-details column, where no two events carry the same keys,
    comes back with every absent key filled in as None unless the column is
    stored as structure rather than as a type.
    """
    df = pd.DataFrame({"details": [{"k": 1}, {"j": "x"}, {}]})
    out = roundtrip(registry, df)
    assert out["details"].tolist() == [{"k": 1}, {"j": "x"}, {}]
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_preserves_nesting_inside_a_cell(registry):
    df = pd.DataFrame({"details": [{"colour": (1, 2, 3), "tags": ["a"], "n": None}]})
    out = roundtrip(registry, df)
    assert out["details"][0] == {"colour": (1, 2, 3), "tags": ["a"], "n": None}


def test_a_cell_json_cannot_carry_leaves_the_frame_uncached(registry):
    """Refused at encode, where wt-task turns it into "not cached"."""
    df = pd.DataFrame({"details": [{"when": object()}]})
    codec = registry.for_value(df)
    with pytest.raises(TypeError):
        codec.encode(df)


def test_arrow_roundtrip_is_not_decided_by_geoarrow_import_order(registry):
    """An extension *name* in field metadata must not become an extension *type*.

    Arrow promotes the one into the other on read when the extension is
    registered, and `geoarrow.pyarrow` registers on import -- so without this
    the same bytes decode to `binary` or to `extension<geoarrow.wkb>` depending
    on what else the process imported.
    """
    import geoarrow.pyarrow  # noqa: F401  # registers geoarrow.* process-wide

    table = arrow_table()
    out = roundtrip(registry, table)
    assert out.schema.field("geometry").type == pa.binary()
    assert out.schema.field("geometry").metadata == {b"ARROW:extension:name": b"geoarrow.wkb"}
    assert out.equals(table)


def test_arrow_roundtrip_of_a_real_geoarrow_extension_type(registry):
    """What the warehouse client actually hands over: a genuine extension type."""
    import geoarrow.pyarrow as ga
    from shapely.geometry import Point as ShapelyPoint

    table = pa.table({"geometry": ga.array([ShapelyPoint(0, 0).wkb, ShapelyPoint(1, 1).wkb])})
    out = roundtrip(registry, table)
    assert out.schema.field("geometry").type == table.schema.field("geometry").type
    assert out.equals(table)
    assert registry.for_value(table).digest(out) == registry.for_value(table).digest(table)
