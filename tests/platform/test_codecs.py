"""Conformance suite for the `wt_task.codecs` entry point.

The headline property is `hit == miss`: whatever a codec stores must come back
indistinguishable from what went in, and the same value must digest the same
however it was encoded. Everything here is one of those two halves.
"""

import json
import warnings
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_serializer
from shapely.geometry import Point
from wt_task.cache import get_registry
from wt_task.skip import SKIP_SENTINEL

from ecoscope.platform import codecs as codecs_module
from ecoscope.platform.codecs import (
    ARROW_CODEC_ID,
    FRAME_CODEC_ID,
    GROUPER_CODEC_ID,
    GROUPERS_CODEC_ID,
    GUARD_CODEC_ID,
    KEYED_CODEC_ID,
    MODEL_CODEC_ID,
    MODELS_CODEC_ID,
    UNDIGESTIBLE,
    digest_dataframe,
)
from ecoscope.platform.indexes import AllGrouper, Month, SpatialGrouper, TemporalGrouper, ValueGrouper, Year
from ecoscope.platform.tasks.filter._filter import TimeRange, TimezoneInfo
from ecoscope.platform.tasks.results._map_utils import TileLayer


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
# groupers
# --------------------------------------------------------------------------- #


def resolved_grouper(regions=None):
    grouper = SpatialGrouper(spatial_index_name="regions")
    grouper.resolve(regions if regions is not None else spatial_regions())
    return grouper


def spatial_regions(name="a"):
    return gpd.GeoDataFrame({"r": [name]}, geometry=[Point(0, 0)], crs="EPSG:4326")


GROUPERS = [
    AllGrouper(),
    ValueGrouper(index_name="event_type"),
    ValueGrouper(index_name="event_type", help_text="pick one"),
    TemporalGrouper(temporal_index=Month()),
    TemporalGrouper(temporal_index=Year()),
    SpatialGrouper(spatial_index_name="regions"),
]


def test_a_bare_all_grouper_is_claimed(registry):
    """`set_groupers` returns `AllGrouper()`, not `[]`, when `groupers` is blank.

    So the ungrouped case needs the single-value codec; without it the whole
    grouped subtree stays uncacheable in exactly the workflows that have no
    groupers at all.
    """
    assert registry.for_value(AllGrouper()).codec_id == GROUPER_CODEC_ID


@pytest.mark.parametrize("grouper", GROUPERS, ids=lambda g: g.index_name)
def test_single_grouper_roundtrips(registry, grouper):
    assert registry.for_value(grouper).codec_id == GROUPER_CODEC_ID
    assert roundtrip(registry, grouper) == grouper


def test_temporal_grouper_roundtrips_its_sort_key(registry):
    """Round-tripped via the directive string; `sort_key` is a bound builtin."""
    grouper = TemporalGrouper(temporal_index=Month())
    back = roundtrip(registry, grouper)
    assert back.temporal_index == grouper.temporal_index
    assert back.sort_key("January") == 1


def test_grouper_list_roundtrips(registry):
    groupers = [ValueGrouper(index_name="event_type"), TemporalGrouper(temporal_index=Month()), resolved_grouper()]
    codec = registry.for_value(groupers)
    assert codec.codec_id == GROUPERS_CODEC_ID

    back = codec.decode(codec.encode(groupers))
    assert back == groupers
    assert back[2].is_resolved
    assert_frame_equal(back[2].spatial_regions, groupers[2].spatial_regions)


def test_an_empty_grouper_list_is_left_to_json(registry):
    """`get_chart_mode_spatial_groupers` returns `[]`, which `json/v1` already round-trips."""
    assert registry.for_value([]).codec_id == "json/v1"
    assert roundtrip(registry, []) == []


def test_a_resolved_spatial_grouper_comes_back_resolved(registry):
    """The hard case: the regions hide in a `PrivateAttr` that `model_dump` drops.

    And `SpatialGrouper` compares equal resolved or not, so `hit == miss` on
    equality alone proves nothing -- `add_spatial_index` would get a grouper
    with nothing to join against.
    """
    regions = spatial_regions()
    grouper = resolved_grouper(regions)
    back = roundtrip(registry, grouper)

    assert back.is_resolved
    assert_frame_equal(back.spatial_regions, regions)
    assert back.spatial_regions.crs == regions.crs


def test_resolved_and_unresolved_spatial_groupers_digest_differently(registry):
    resolved = resolved_grouper()
    unresolved = SpatialGrouper(spatial_index_name="regions")
    codec = registry.for_value(resolved)

    assert resolved == unresolved, "the hazard itself: equality sees only the name"
    assert codec.digest(resolved) != codec.digest(unresolved)
    assert codec.digest(resolved) != codec.digest(resolved_grouper(spatial_regions("b")))


def test_grouper_digest_is_stable_across_encodings(registry):
    """Two encodes of the same resolved grouper, and the value one of them read back."""
    grouper = resolved_grouper()
    codec = registry.for_value(grouper)
    codec.encode(grouper)
    assert codec.digest(grouper) == codec.digest(grouper) == codec.digest(roundtrip(registry, grouper))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (ValueGrouper(index_name="x"), ValueGrouper(index_name="y")),
        (ValueGrouper(index_name="x"), ValueGrouper(index_name="x", help_text="h")),
        (TemporalGrouper(temporal_index=Month()), TemporalGrouper(temporal_index=Year())),
        (TemporalGrouper(temporal_index=Month()), TemporalGrouper(temporal_index=Month(), help_text="h")),
        (SpatialGrouper(spatial_index_name="a"), SpatialGrouper(spatial_index_name="b")),
        (AllGrouper(), AllGrouper(index_name="Everything")),
        # Two groupers whose only difference is their type: both call it "x",
        # and `index_name` would make them look the same to a digest over it.
        (ValueGrouper(index_name="x"), AllGrouper(index_name="x")),
    ],
)
def test_grouper_digest_is_sensitive_to_every_field(registry, a, b):
    assert registry.for_value(a).digest(a) != registry.for_value(b).digest(b)


def test_a_grouper_list_digest_is_sensitive_to_order_and_membership(registry):
    groupers = [ValueGrouper(index_name="x"), TemporalGrouper(temporal_index=Month())]
    codec = registry.for_value(groupers)
    assert codec.digest(groupers) != codec.digest(list(reversed(groupers)))
    assert codec.digest(groupers) != codec.digest(groupers[:1])


def test_a_grouper_inside_a_keyed_iterable_is_carried_whole(registry):
    """Keyed values dispatch through the registry, so this works by construction.

    Worth pinning anyway: before the grouper codec existed the whole keyed
    iterable was refused, which is the shape `gather_dashboard`'s inputs take.
    """
    value = [((("All", "=", "True"),), [resolved_grouper()])]
    codec = registry.for_value(value)
    assert codec.codec_id == KEYED_CODEC_ID
    assert codec.decode(codec.encode(value))[0][1][0].is_resolved


def test_a_list_mixing_groupers_and_other_values_is_not_claimed(registry):
    assert registry.for_value([ValueGrouper(index_name="x"), "x"]).codec_id == GUARD_CODEC_ID


def test_a_grouper_gaining_a_field_declines_rather_than_dropping_it(registry, monkeypatch):
    """Fail closed: the field list is pinned, so a new field is a decline.

    Encoding it regardless would leave the new field out of both payload and
    digest, and two groupers differing only in it would share a cache entry.
    """
    monkeypatch.setitem(codecs_module._GROUPER_FIELDS, "ValueGrouper", ("index_name",))
    assert not codecs_module.is_grouper(ValueGrouper(index_name="x"))
    assert registry.for_value([ValueGrouper(index_name="x")]).codec_id == GUARD_CODEC_ID


def test_a_grouper_holding_an_undigestible_frame_is_refused(registry):
    """A leaf its own codec cannot identify must take the container down with it.

    `UNDIGESTIBLE` is one shared marker, so standing it in for the frame would
    make two differently-resolved groupers digest identically.
    """

    class Unreprable:
        def __repr__(self):
            raise RuntimeError("boom")

    regions = gpd.GeoDataFrame({"r": [Unreprable()]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    grouper = resolved_grouper(regions)
    codec = registry.for_value(grouper)

    assert codec.digest(grouper) == UNDIGESTIBLE
    with pytest.raises(TypeError):
        codec.encode(grouper)


# --------------------------------------------------------------------------- #
# pydantic models
# --------------------------------------------------------------------------- #


class Plain(BaseModel):
    name: str
    count: int = 0


class Excluding(BaseModel):
    name: str
    note: str = Field(default="", exclude=True)


class Private(BaseModel):
    name: str
    _cached: object | None = PrivateAttr(default=None)


class Reshaping(BaseModel):
    name: str

    @model_serializer
    def _ser(self):
        return {"renamed": self.name}


class Loose(BaseModel):
    """A field whose annotation cannot carry a tuple back."""

    payload: Any = None


def timezone_info():
    return TimezoneInfo(label="Nairobi", tzCode="Africa/Nairobi", name="(UTC+03:00) Nairobi", utc="+03:00")


def time_range(since="2015-01-01T00:00:00"):
    return TimeRange(since=since, until="2015-12-31T23:59:59", timezone=timezone_info())


def test_config_models_roundtrip(registry):
    """`time_range` and `get_timezone` gate the whole events frame chain."""
    for value in (timezone_info(), time_range()):
        assert registry.for_value(value).codec_id == MODEL_CODEC_ID
        assert roundtrip(registry, value) == value


def test_a_model_list_roundtrips(registry):
    value = [timezone_info(), time_range()]
    codec = registry.for_value(value)
    assert codec.codec_id == MODELS_CODEC_ID
    assert codec.decode(codec.encode(value)) == value


def test_model_digest_is_stable_and_sensitive(registry):
    value = time_range()
    codec = registry.for_value(value)
    assert codec.digest(value) == codec.digest(time_range()) == codec.digest(roundtrip(registry, value))
    assert codec.digest(value) != codec.digest(time_range(since="2016-01-01T00:00:00"))
    assert codec.digest(timezone_info()) != codec.digest(
        TimezoneInfo(label="UTC", tzCode="UTC", name="UTC", utc="+00:00")
    )


def test_two_models_dumping_alike_digest_differently(registry):
    """The class is part of the identity; the dump alone is not."""

    class Twin(BaseModel):
        name: str
        count: int = 0

    a, b = Plain(name="x"), Twin(name="x")
    assert a.model_dump() == b.model_dump()
    assert registry.for_value(a).digest(a) != registry.for_value(b).digest(b)


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (Excluding(name="x"), "an excluded field is dropped by the serializer"),
        (Private(name="x"), "a private attribute is invisible to model_dump"),
        (Reshaping(name="x"), "a model serializer emits a shape model_validate cannot read"),
        (Loose(payload=(1, 2)), "a tuple under `Any` comes back a list"),
    ],
)
def test_models_that_cannot_round_trip_are_declined(registry, value, why):
    assert registry.for_value(value) is None, why


def test_a_model_that_rejects_its_own_dump_is_declined(registry):
    """What no structural check could see, and why the predicate pays for a dump.

    `TileLayer` used to be this case -- it dumped `layer_name=""` and its own
    validator refused it back -- which is now fixed at the model.
    """

    class Picky(BaseModel):
        name: str = ""

        @field_validator("name")
        def _reject_blank(cls, v):
            if not v:
                raise ValueError("no")
            return v

    assert registry.for_value(Picky.model_construct(name="")) is None


@pytest.mark.parametrize("layer_name", ["", "TERRAIN"])
def test_tile_layers_roundtrip_singly_and_in_a_list(registry, layer_name):
    """`base_map_defs` returns `list[TileLayer]` -- the last refused config step."""
    custom = TileLayer(url="https://example.com/{z}/{x}/{y}.png", opacity=0.5, max_zoom=17)
    layer = TileLayer(layer_name=layer_name) if layer_name else custom

    assert registry.for_value(layer).codec_id == MODEL_CODEC_ID
    assert roundtrip(registry, layer) == layer

    # The real `base_map_defs` shape: a preset under a semi-transparent custom one.
    layers = [layer, TileLayer(layer_name="SATELLITE", opacity=0.5)]
    codec = registry.for_value(layers)
    assert codec.codec_id == MODELS_CODEC_ID
    assert codec.decode(codec.encode(layers)) == layers
    assert codec.digest(layers) != codec.digest(list(reversed(layers)))


def test_a_list_of_models_that_cannot_round_trip_still_reaches_the_guard(registry):
    """One unclaimable element disqualifies the list; it must not half-encode."""
    value = [Plain(name="ok"), Reshaping(name="bad")]
    codec = registry.for_value(value)
    assert codec.codec_id == GUARD_CODEC_ID
    assert codec.digest(value) == UNDIGESTIBLE
    with pytest.raises(TypeError):
        codec.encode(value)


def test_groupers_are_claimed_by_their_own_codec_not_the_model_one(registry):
    """`AllGrouper` would pass the model predicate; MRO must keep it out.

    The others are declined by it outright -- `help_text` is `exclude=True` and
    `SpatialGrouper`'s regions are a private attribute -- which is the same
    fail-closed rule arriving at the same answer from the other side.
    """
    assert registry.for_value(AllGrouper()).codec_id == GROUPER_CODEC_ID
    assert registry.for_value([AllGrouper()]).codec_id == GROUPERS_CODEC_ID
    assert not codecs_module.is_roundtrippable_model(ValueGrouper(index_name="x"))
    assert not codecs_module.is_roundtrippable_model(SpatialGrouper(spatial_index_name="r"))


def test_decode_refuses_a_class_from_an_unimported_package(registry):
    """Importing a module is code execution, and the path came out of the store."""
    payload = json.dumps({"type": "not_a_real_package.mod:Thing", "data": {}}).encode()
    with pytest.raises(ValueError, match="has not imported"):
        registry.for_id(MODEL_CODEC_ID).decode(payload)


# --------------------------------------------------------------------------- #
# the JSON guard
# --------------------------------------------------------------------------- #


def test_a_list_json_cannot_represent_is_refused_rather_than_stringified(registry):
    """`json/v1` encodes with `default=str`, so a value it cannot carry is stringified.

    Nothing downstream could tell that from the real thing on a hit, so the
    guard turns it into "not cached" instead.
    """
    value = [{1: "a"}]
    codec = registry.for_value(value)
    assert codec.codec_id == GUARD_CODEC_ID
    assert codec.digest(value) == UNDIGESTIBLE
    with pytest.raises(TypeError):
        codec.encode(value)


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
