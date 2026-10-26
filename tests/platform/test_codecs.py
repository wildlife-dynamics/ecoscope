"""Conformance suite for the leaf codecs `ecoscope.platform` declares.

The headline property is `hit == miss`: whatever a codec stores must come back
indistinguishable from what went in, and the same value must digest the same
however it was encoded. Everything here is one of those two halves.

Which codec applies is no longer decided by inspecting a value, so the suite
starts from the declarations: if an alias or a class stopped declaring one,
every round-trip below would still pass while nothing cached.
"""

import datetime
import importlib
import json
import pkgutil
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal
from pydantic import BaseModel
from shapely.geometry import Point
from wt_contracts import Cacheable, declared_structure
from wt_task.cache import Tree, Undigestible, resolve_structure
from wt_task.skip import SKIP_SENTINEL

import ecoscope.platform
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.codecs import (
    ARROW,
    ARROW_CODEC_ID,
    FRAME,
    FRAME_CODEC_ID,
    KEYED_ANY,
    digest_dataframe,
    grouper_codec,
    model_codec,
)
from ecoscope.platform.indexes import (
    AllGrouper,
    CombinedKeyedIterable,
    KeyedIterableOfAny,
    Month,
    SpatialGrouper,
    TemporalGrouper,
    UserDefinedGroupers,
    ValueGrouper,
    Year,
)
from ecoscope.platform.tasks.filter._filter import TimeRange, TimezoneInfo
from ecoscope.platform.tasks.results._map_utils import TileLayer


def roundtrip(codec, value):
    return codec.decode(codec.encode(value))


# --------------------------------------------------------------------------- #
# declarations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("annotation", "ref"),
    [
        (AnyDataFrame, "ecoscope.platform.codecs:FRAME"),
        (KeyedIterableOfAny, "ecoscope.platform.codecs:KEYED_ANY"),
        (CombinedKeyedIterable, "ecoscope.platform.codecs:KEYED_ANY"),
    ],
)
def test_the_aliases_declare_their_codec(annotation, ref):
    """One alias edit stands behind every task that returns through it.

    A missing marker is invisible from a round-trip test -- the codec still
    works, nothing reaches it -- so the declaration is asserted directly.
    """
    assert declared_structure(annotation) == {"k": "ref", "ref": ref, "type": None}


def test_the_grouper_alias_resolves_arm_by_arm():
    """`UserDefinedGroupers` declares nothing itself; each class does.

    So the list is structure wt walks rather than one opaque leaf, and a
    grouper that stopped declaring shows up as a `None` arm, not as a list
    codec quietly encoding it some other way.
    """
    structure = declared_structure(UserDefinedGroupers)
    assert structure["k"] == "list"
    assert [arm["type"] for arm in structure["of"]["of"]] == [
        "ecoscope.platform.indexes:ValueGrouper",
        "ecoscope.platform.indexes:TemporalGrouper",
        "ecoscope.platform.indexes:SpatialGrouper",
    ]
    assert {arm["ref"] for arm in structure["of"]["of"]} == {"ecoscope.platform.codecs:grouper_codec"}


def test_a_frame_alias_resolves_through_the_declaration():
    """`declared_structure` must reach `FRAME` through pandera's subscripted alias."""
    from ecoscope.platform.annotations import AnyGeoDataFrame, EmptyDataFrame

    for alias in (AnyDataFrame, AnyGeoDataFrame, EmptyDataFrame):
        assert declared_structure(alias)["ref"] == "ecoscope.platform.codecs:FRAME"


def test_every_declared_codec_ref_imports():
    """A ref is a string until something resolves it; a typo would surface at run time."""
    for annotation in (AnyDataFrame, KeyedIterableOfAny, UserDefinedGroupers, AllGrouper, TimeRange):
        assert resolve_structure(declared_structure(annotation)) is not None


# --------------------------------------------------------------------------- #
# frame round-trips -- one per fidelity case
# --------------------------------------------------------------------------- #


def test_roundtrip_preserves_a_non_default_crs():
    gdf = gpd.GeoDataFrame({"a": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:3857")
    out = roundtrip(FRAME, gdf)
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == gdf.crs
    assert out.equals(gdf)


def test_roundtrip_preserves_a_multiindex():
    """`merge_df` sets the index to composite filter keys, which `index=False` drops."""
    index = pd.MultiIndex.from_tuples([("a", "1"), ("b", "2")], names=["grouper", "value"])
    df = pd.DataFrame({"x": [1, 2]}, index=index)
    out = roundtrip(FRAME, df)
    assert list(out.index.names) == ["grouper", "value"]
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_preserves_tz_aware_timestamps():
    df = pd.DataFrame({"t": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])})
    out = roundtrip(FRAME, df)
    assert out["t"].dtype == df["t"].dtype
    pd.testing.assert_frame_equal(out, df)


@pytest.mark.parametrize(
    "tz",
    [
        datetime.timezone(datetime.timedelta(hours=3)),
        datetime.timezone(datetime.timedelta(hours=-5, minutes=-30), "Somewhere"),
        datetime.timezone.utc,
    ],
    ids=["offset", "named-offset", "utc"],
)
def test_roundtrip_preserves_a_fixed_offset_timezone(tz):
    """What `convert_values_to_timezone` makes of a time range's `utc` offset.

    Parquet keeps only the offset and pandas reads back a `pytz.FixedOffset`,
    whose dtype prints differently -- so without restoring it, every step
    computed from a hit digests differently from one computed from a miss.
    """
    df = pd.DataFrame({"t": pd.to_datetime(["2015-01-10T00:00:00Z", "2015-01-11T00:00:00Z"]).tz_convert(tz)})
    out = roundtrip(FRAME, df)
    assert str(out["t"].dtype) == str(df["t"].dtype)
    assert out["t"].dt.tz == tz
    assert digest_dataframe(out) == digest_dataframe(df)


def test_roundtrip_preserves_a_column_mixing_ints_and_strings():
    """`concat_dataframes` of trajectories and events: Arrow refuses `1` beside `"1"`."""
    df = pd.DataFrame({"serial": [23456, "23457", None]})
    out = roundtrip(FRAME, df)
    assert out["serial"].tolist() == [23456, "23457", None]


def test_roundtrip_keeps_ints_beside_floats_as_ints():
    df = pd.DataFrame({"n": pd.Series([1, 1.5], dtype=object)})
    out = roundtrip(FRAME, df)
    assert [type(v) for v in out["n"]] == [int, float]


def test_a_mixed_column_does_not_digest_like_its_strings():
    """pandas hashes object columns through `str`, which would make these collide."""
    mixed = pd.DataFrame({"serial": [1, "x"]})
    strings = pd.DataFrame({"serial": ["1", "x"]})
    assert digest_dataframe(mixed) != digest_dataframe(strings)


def test_roundtrip_preserves_dict_cells():
    """ER event details arrive as dicts in an object column."""
    df = pd.DataFrame({"details": [{"k": 1}, {"k": 2, "j": "x"}]})
    out = roundtrip(FRAME, df)
    assert isinstance(out["details"][0], dict)
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_preserves_tuple_cells():
    """RGBA colours are tuples, and parquet hands them back as `ndarray`.

    The inverse hazard is the reason decode reads a recorded spec rather than
    guessing: restoring every sequence as a tuple would corrupt genuine lists.
    """
    df = pd.DataFrame({"rgba": [(1, 2, 3, 255), (4, 5, 6, 255)]})
    out = roundtrip(FRAME, df)
    assert out["rgba"].tolist() == [(1, 2, 3, 255), (4, 5, 6, 255)]
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_keeps_lists_and_tuples_apart_in_the_same_frame():
    df = pd.DataFrame({"t": [(1, 2), (3, 4)], "l": [[1, 2], [3, 4]]})
    out = roundtrip(FRAME, df)
    assert out["t"].tolist() == [(1, 2), (3, 4)]
    assert out["l"].tolist() == [[1, 2], [3, 4]]


def test_roundtrip_handles_a_column_mixing_lists_and_tuples():
    df = pd.DataFrame({"mixed": [(1, 2), [3, 4], None]})
    out = roundtrip(FRAME, df)
    assert out["mixed"][0] == (1, 2)
    assert out["mixed"][1] == [3, 4]
    assert out["mixed"][2] is None


def test_roundtrip_of_an_empty_geodataframe():
    """`EmptyDataFrame` is a real return type, not just an edge case."""
    gdf = gpd.GeoDataFrame(
        {"a": pd.Series([], dtype="int64")},
        geometry=gpd.GeoSeries([], dtype="geometry"),
        crs="EPSG:4326",
    )
    out = roundtrip(FRAME, gdf)
    assert len(out) == 0
    assert out.crs == gdf.crs


def test_roundtrip_of_a_frame_carrying_attrs():
    df = pd.DataFrame({"a": [1]})
    df.attrs = {"source": "warehouse"}
    out = roundtrip(FRAME, df)
    assert out.attrs == {"source": "warehouse"}


def test_roundtrip_of_a_frame_without_geometry_returns_a_plain_dataframe():
    df = pd.DataFrame({"a": [1, 2]})
    out = roundtrip(FRAME, df)
    assert type(out) is pd.DataFrame
    pd.testing.assert_frame_equal(out, df)


# --------------------------------------------------------------------------- #
# frame digest -- stability and sensitivity
# --------------------------------------------------------------------------- #


def test_digest_is_stable_while_the_encoded_bytes_need_not_be():
    """The whole reason `digest` is a separate callable from `encode`.

    Parquet is not byte-stable in general; a digest derived from it would
    invalidate every downstream step on every run.
    """
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert FRAME.digest(df) == FRAME.digest(df.copy())
    assert FRAME.digest(df) == digest_dataframe(df)


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


def test_digest_of_a_frame_holding_a_resolved_spatial_grouper():
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


class Unreprable:
    """A cell pandas can neither hash nor `repr`."""

    def __repr__(self):
        raise RuntimeError("boom")


def test_digest_declines_rather_than_raising_on_an_unhashable_value():
    """`digest_dataframe` raises -- correct for a filename, fatal for a run.

    The codec turns it into `Undigestible`, which wt-task reads as "this step
    is not cached" rather than as a failure.
    """

    df = pd.DataFrame({"bad": [{"k": Unreprable()}]})
    with pytest.raises(TypeError):
        digest_dataframe(df)
    with pytest.raises(Undigestible):
        FRAME.digest(df)


# --------------------------------------------------------------------------- #
# keyed iterables
# --------------------------------------------------------------------------- #


def test_keyed_iterable_roundtrips_keys_as_tuples():
    """A key is a dict key downstream, so it must not come back a list."""
    df = pd.DataFrame({"a": [1, 2]})
    value = [
        ((("month", "=", "January"),), df),
        ((("month", "=", "February"),), df.iloc[:1]),
    ]
    out = roundtrip(KEYED_ANY, value)
    assert [k for k, _ in out] == [k for k, _ in value]
    assert all(isinstance(k, tuple) and isinstance(k[0], tuple) for k, _ in out)
    pd.testing.assert_frame_equal(out[0][1], df)
    # The property the tuple-ness is for.
    assert {k for k, _ in out} == {k for k, _ in value}


def test_keyed_iterable_roundtrips_combined_values():
    """`groupbykey` nests a list of values under each key."""
    df = pd.DataFrame({"a": [1]})
    value = [((("x", "=", "1"),), [df, "a path", 3])]
    out = roundtrip(KEYED_ANY, value)
    pd.testing.assert_frame_equal(out[0][1][0], df)
    assert out[0][1][1:] == ["a path", 3]


def test_keyed_iterable_roundtrips_a_skip_sentinel():
    """`groupbykey_passthrough_skip` keeps the key and collapses values to a sentinel."""
    value = [((("x", "=", "1"),), SKIP_SENTINEL)]
    assert roundtrip(KEYED_ANY, value)[0][1] is SKIP_SENTINEL


def test_keyed_iterable_roundtrips_a_null_key():
    df = pd.DataFrame({"a": [1]})
    assert roundtrip(KEYED_ANY, [(None, df)])[0][0] is None


def test_keyed_iterable_digest_is_stable_and_sensitive():
    df = pd.DataFrame({"a": [1, 2]})
    same = KEYED_ANY.digest([((("x", "=", "1"),), df.copy())])
    assert KEYED_ANY.digest([((("x", "=", "1"),), df)]) == same
    assert KEYED_ANY.digest([((("x", "=", "2"),), df)]) != same
    assert KEYED_ANY.digest([((("x", "=", "1"),), df.iloc[:1])]) != same


def test_keyed_iterable_refuses_a_value_no_arm_accepts():
    """Refused, never stringified: a repr served as the value is worse than a miss."""
    value = [((("x", "=", "1"),), object())]
    with pytest.raises(Undigestible):
        KEYED_ANY.digest(value)
    with pytest.raises(Undigestible):
        KEYED_ANY.encode(value)


def test_keyed_iterable_carries_an_arrow_table():
    """`persist_geoarrow_for_pydeck` puts tables in the keyed position."""
    table = pa.table({"a": [1, 2]})
    assert roundtrip(KEYED_ANY, [((("x", "=", "1"),), table)])[0][1].equals(table)


# --------------------------------------------------------------------------- #
# groupers
# --------------------------------------------------------------------------- #


def spatial_regions(name="a"):
    return gpd.GeoDataFrame({"r": [name]}, geometry=[Point(0, 0)], crs="EPSG:4326")


def resolved_grouper(regions=None):
    grouper = SpatialGrouper(spatial_index_name="regions")
    grouper.resolve(regions if regions is not None else spatial_regions())
    return grouper


USER_GROUPERS = [
    ValueGrouper(index_name="event_type"),
    ValueGrouper(index_name="event_type", help_text="pick one"),
    TemporalGrouper(temporal_index=Month()),
    TemporalGrouper(temporal_index=Year()),
    SpatialGrouper(spatial_index_name="regions"),
]


@pytest.mark.parametrize("grouper", USER_GROUPERS, ids=lambda g: g.index_name)
def test_single_grouper_roundtrips(grouper):
    assert roundtrip(grouper_codec(type(grouper)), grouper) == grouper


def test_temporal_grouper_roundtrips_its_sort_key():
    """Round-tripped via the directive string; `sort_key` is a bound builtin."""
    grouper = TemporalGrouper(temporal_index=Month())
    back = roundtrip(grouper_codec(TemporalGrouper), grouper)
    assert back.temporal_index == grouper.temporal_index
    assert back.sort_key("January") == 1


def grouper_list_codec():
    """The tree wt builds for `UserDefinedGroupers`, from the declaration alone."""
    return resolve_structure(declared_structure(UserDefinedGroupers))


def test_grouper_list_roundtrips():
    groupers = [ValueGrouper(index_name="event_type"), TemporalGrouper(temporal_index=Month()), resolved_grouper()]
    back = roundtrip(grouper_list_codec(), groupers)
    assert back == groupers
    assert back[2].is_resolved
    assert_frame_equal(back[2].spatial_regions, groupers[2].spatial_regions)


def test_an_empty_grouper_list_roundtrips():
    """`get_chart_mode_spatial_groupers` returns `[]` when the mode is not spatial."""
    assert roundtrip(grouper_list_codec(), []) == []


def test_a_resolved_spatial_grouper_comes_back_resolved():
    """The hard case: the regions hide in a `PrivateAttr` that `model_dump` drops.

    And `SpatialGrouper` compares equal resolved or not, so `hit == miss` on
    equality alone proves nothing -- `add_spatial_index` would get a grouper
    with nothing to join against.
    """
    regions = spatial_regions()
    back = roundtrip(grouper_codec(SpatialGrouper), resolved_grouper(regions))

    assert back.is_resolved
    assert_frame_equal(back.spatial_regions, regions)
    assert back.spatial_regions.crs == regions.crs


def test_resolved_and_unresolved_spatial_groupers_digest_differently():
    codec = grouper_codec(SpatialGrouper)
    resolved = resolved_grouper()
    unresolved = SpatialGrouper(spatial_index_name="regions")

    assert resolved == unresolved, "the hazard itself: equality sees only the name"
    assert codec.digest(resolved) != codec.digest(unresolved)
    assert codec.digest(resolved) != codec.digest(resolved_grouper(spatial_regions("b")))


def test_grouper_digest_is_stable_across_encodings():
    """Two encodes of the same resolved grouper, and the value one of them read back."""
    codec = grouper_codec(SpatialGrouper)
    grouper = resolved_grouper()
    codec.encode(grouper)
    assert codec.digest(grouper) == codec.digest(grouper) == codec.digest(roundtrip(codec, grouper))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (ValueGrouper(index_name="x"), ValueGrouper(index_name="y")),
        (ValueGrouper(index_name="x"), ValueGrouper(index_name="x", help_text="h")),
        (TemporalGrouper(temporal_index=Month()), TemporalGrouper(temporal_index=Year())),
        (TemporalGrouper(temporal_index=Month()), TemporalGrouper(temporal_index=Month(), help_text="h")),
        (SpatialGrouper(spatial_index_name="a"), SpatialGrouper(spatial_index_name="b")),
    ],
)
def test_grouper_digest_is_sensitive_to_every_field(a, b):
    assert grouper_codec(type(a)).digest(a) != grouper_codec(type(b)).digest(b)


def test_two_grouper_types_sharing_an_index_name_digest_differently():
    """`index_name` alone would make a `ValueGrouper` and an `AllGrouper` look alike."""
    value, every = ValueGrouper(index_name="x"), AllGrouper(index_name="x")
    assert grouper_codec(ValueGrouper).digest(value) != model_codec(AllGrouper).digest(every)


def test_a_grouper_list_digest_is_sensitive_to_order_and_membership():
    groupers = [ValueGrouper(index_name="x"), TemporalGrouper(temporal_index=Month())]
    codec = grouper_list_codec()
    assert codec.digest(groupers) != codec.digest(list(reversed(groupers)))
    assert codec.digest(groupers) != codec.digest(groupers[:1])


def test_a_grouper_gaining_a_field_declines_rather_than_dropping_it(monkeypatch):
    """Fail closed: the field list is pinned, so a new field is a decline.

    Encoding it regardless would leave the new field out of both payload and
    digest, and two groupers differing only in it would share a cache entry.
    """
    from ecoscope.platform import codecs as codecs_module

    monkeypatch.setitem(codecs_module._GROUPER_FIELDS, "ValueGrouper", ("index_name",))
    with pytest.raises(Undigestible):
        grouper_codec(ValueGrouper).digest(ValueGrouper(index_name="x"))


def test_a_grouper_holding_an_undigestible_frame_is_refused():
    """A leaf its own codec cannot identify must take the container down with it."""
    regions = gpd.GeoDataFrame({"r": [{"k": Unreprable()}]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    codec = grouper_codec(SpatialGrouper)
    with pytest.raises(Undigestible):
        codec.digest(resolved_grouper(regions))


def test_a_grouper_codec_refuses_another_grouper_class():
    """`exact`, so a payload can never be decoded back into the wrong class."""
    tree = Tree({"k": "ref", "codec": grouper_codec(ValueGrouper)})
    with pytest.raises(Undigestible):
        tree.digest(TemporalGrouper(temporal_index=Month()))


# --------------------------------------------------------------------------- #
# pydantic models
# --------------------------------------------------------------------------- #


def timezone_info():
    return TimezoneInfo(label="Nairobi", tzCode="Africa/Nairobi", name="(UTC+03:00) Nairobi", utc="+03:00")


def time_range(since="2015-01-01T00:00:00"):
    return TimeRange(since=since, until="2015-12-31T23:59:59", timezone=timezone_info())


# The structural half of the old `is_roundtrippable_model` is now checked once,
# at declaration. This is the behavioural half -- a value that survives the dump
# but comes back as something else -- which only an instance can show, so every
# declared model gets one. Classes whose fields are all optional build
# themselves; the rest need their required fields named here, which is also what
# makes a new model with required fields fail until someone covers it.
REQUIRED_FIELDS: dict[str, dict] = {
    "AllGrouper": {"index_name": "Everything"},
    "_BaseSummaryParam": {"display_name": "Total"},
    "StatSummaryParam": {"display_name": "Total", "aggregator": "sum", "column": "distance"},
    "UniqueSummaryParam": {"display_name": "Species", "aggregator": "unique", "column": "species"},
    "CoverageSummaryParam": {"display_name": "Covered", "aggregator": "coverage_area"},
    "NightDayRatioSummaryParam": {"display_name": "Night/day", "aggregator": "night_day_ratio"},
    "RatioSummaryParam": {
        "display_name": "Rate",
        "aggregator": "ratio",
        "numerator": {"aggregator": "count", "column": "events"},
        "denominator": {"aggregator": "sum", "column": "distance"},
    },
    "BbmmMethodArgs": {"location_error_settings": {}},
    "EtdMethodArgs": {"speed_settings": {}},
    "TimezoneInfo": {"label": "Nairobi", "tzCode": "Africa/Nairobi", "name": "(UTC+03:00) Nairobi", "utc": "+03:00"},
    "TimeRange": {
        "since": "2015-01-01T00:00:00",
        "until": "2015-12-31T23:59:59",
        "timezone": {"label": "UTC", "tzCode": "UTC", "name": "UTC", "utc": "+00:00"},
    },
    "ChartFigure": {"metric": {"display_name": "Total", "aggregator": "sum", "column": "distance"}},
    "OutputFiles": {"files": ["/tmp/a.csv", [((("month", "=", "January"),), "/tmp/b.csv")]]},
    "BitmapLayerDefinition": {"image": "https://example.com/x.png", "bounds": [0.0, 0.0, 1.0, 1.0]},
    "DeckJsonSpec": {
        "layers": [{"@@type": "ScatterplotLayer"}],
        "initialViewState": {"latitude": 0.0, "longitude": 0.0, "zoom": 3},
        "views": [{"@@type": "MapView"}],
    },
    "Coordinate": {"x": 1.5, "y": -2.5},
    "TileLayer": {"layer_name": "TERRAIN"},
    "CustomMetric": {"display_name": "Total", "aggregator": "sum", "column": "distance"},
}


def _declared_models() -> dict[str, type]:
    """Every pydantic model under `ecoscope.platform` declaring a codec."""
    found: dict[str, type] = {}
    for info in pkgutil.walk_packages(ecoscope.platform.__path__, "ecoscope.platform."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseModel) and issubclass(obj, Cacheable):
                found[f"{obj.__module__}:{obj.__qualname__}"] = obj
    return dict(sorted(found.items()))


# Split by which codec a model declares. The groupers encode themselves, so
# putting them through `model_codec` below would prove the wrong thing -- their
# round trip is covered in the grouper section above.
DECLARED_MODELS = [cls for cls in _declared_models().values() if not cls.__wt_codec_owns_round_trip__]
SELF_ENCODING_MODELS = [cls for cls in _declared_models().values() if cls.__wt_codec_owns_round_trip__]


def instance(cls: type[BaseModel]) -> BaseModel:
    return cls(**REQUIRED_FIELDS.get(cls.__qualname__, {}))


def test_the_declared_models_are_the_ones_the_plan_names():
    """A model losing its declaration is invisible from a round-trip test."""
    names = {cls.__qualname__ for cls in DECLARED_MODELS}
    assert {
        "AllGrouper",
        "TimeRange",
        "TimezoneInfo",
        "DeckJsonSpec",
        "TileLayer",
        "ChartConfig",
        "BarLayoutStyle",
    } <= names
    # Undeclared on purpose, each for a reason no structural check can see.
    # Declaring one would cache a value its own validators hand back changed.
    undeclarable = {
        "Dashboard": "a model_serializer emits a shape model_validate cannot read",
        "GamTrendModel": "excludes `bounds_settings` from its dump",
    }
    assert not names & set(undeclarable), f"declared despite not round-tripping: {names & set(undeclarable)}"


def test_only_the_groupers_own_their_round_trip():
    """The opt-out skips the structural check, so it must stay rare and deliberate.

    A model that sets it without a codec that really does encode it would be
    cached through `model_dump` with no check that the dump survives.
    """
    assert {cls.__qualname__ for cls in SELF_ENCODING_MODELS} == {
        "ValueGrouper",
        "TemporalGrouper",
        "SpatialGrouper",
    }
    assert {cls for cls in SELF_ENCODING_MODELS} == {type(g) for g in USER_GROUPERS}


@pytest.mark.parametrize("cls", DECLARED_MODELS, ids=lambda c: c.__qualname__)
def test_cacheable_model_round_trips(cls):
    """`hit == miss` for every declared model, through its own codec."""
    value = instance(cls)
    codec = model_codec(cls)
    back = roundtrip(codec, value)
    assert back == value
    assert back.model_dump() == value.model_dump()
    assert codec.digest(back) == codec.digest(value)


def test_model_digest_is_stable_and_sensitive():
    codec = model_codec(TimeRange)
    value = time_range()
    assert codec.digest(value) == codec.digest(time_range()) == codec.digest(roundtrip(codec, value))
    assert codec.digest(value) != codec.digest(time_range(since="2016-01-01T00:00:00"))

    tz = model_codec(TimezoneInfo)
    assert tz.digest(timezone_info()) != tz.digest(TimezoneInfo(label="UTC", tzCode="UTC", name="UTC", utc="+00:00"))


def test_a_time_range_read_back_formats_its_times_identically():
    """The dashboard prints `since`/`until` with `%Z`.

    pydantic reads a dumped offset back as its own `TzInfo`, which spells
    `+03:00` where the `datetime.timezone` it was built with said `UTC+03:00`:
    equal instants and an equal digest, but a different dashboard label.
    """
    codec = model_codec(TimeRange)
    value = time_range()
    back = roundtrip(codec, value)
    fmt = "%d %b %Y %H:%M:%S %Z"
    assert back.since.strftime(fmt) == value.since.strftime(fmt)
    assert back.until.strftime(fmt) == value.until.strftime(fmt)


def test_two_models_dumping_alike_do_not_share_a_codec_id():
    """The class is part of the identity, and the payload no longer carries it.

    So the codec id has to, or a declaration that moved would decode one
    model's bytes into another model.
    """

    class A(BaseModel):
        name: str

    class B(BaseModel):
        name: str

    a, b = A(name="x"), B(name="x")
    assert a.model_dump() == b.model_dump()
    assert model_codec(A).codec_id != model_codec(B).codec_id


def test_a_model_codec_refuses_a_subclass():
    """`exact`: encoding a subclass through its parent would narrow it silently."""

    class Parent(BaseModel):
        name: str

    class Child(Parent):
        extra: int = 0

    tree = Tree({"k": "ref", "codec": model_codec(Parent)})
    assert tree.digest(Parent(name="x"))
    with pytest.raises(Undigestible):
        tree.digest(Child(name="x", extra=1))


def test_a_model_that_cannot_dump_losslessly_declines():
    """`warnings="error"`: pydantic's default is to stringify and carry on."""

    class Loose(BaseModel):
        payload: Any = None

    with pytest.raises(Undigestible):
        model_codec(Loose).digest(Loose(payload=object()))


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


def test_arrow_table_roundtrips_through_ipc_with_metadata_intact():
    table = arrow_table()
    out = roundtrip(ARROW, table)
    assert out.equals(table)
    assert out.schema.field("count").metadata == {b"role": b"measure"}
    assert out.schema.field("geometry").metadata == {b"ARROW:extension:name": b"geoarrow.wkb"}
    assert out.schema.metadata == {b"ecoscope": b"v1"}


def test_arrow_digest_is_stable_across_encodings_and_chunking():
    """IPC bytes move with chunk layout; identity must not."""
    table = arrow_table()
    chunked = pa.concat_tables([table.slice(0, 1), table.slice(1, 1)])
    assert ARROW.digest(table) == ARROW.digest(arrow_table())
    assert ARROW.digest(chunked) == ARROW.digest(table)
    assert ARROW.encode(chunked) != ARROW.encode(table)


def test_arrow_digest_is_sensitive_to_values_and_to_field_metadata():
    table = arrow_table()
    other_values = pa.table([pa.array([1, 3]), pa.array([b"a", b"b"])], schema=table.schema)
    stripped = table.replace_schema_metadata(None)
    assert ARROW.digest(other_values) != ARROW.digest(table)
    assert ARROW.digest(stripped) != ARROW.digest(table)


def test_the_two_frame_codecs_do_not_collide():
    """Namespaced by codec id, so the same logical data does not dedup across formats.

    The price of not defining a canonical form across two type systems.
    """
    df = pd.DataFrame({"a": [1, 2]})
    table = pa.Table.from_pandas(df, preserve_index=False)
    assert FRAME.codec_id == FRAME_CODEC_ID != ARROW_CODEC_ID == ARROW.codec_id
    assert FRAME.digest(df) != ARROW.digest(table)


# --------------------------------------------------------------------------- #
# the shape of the payload
# --------------------------------------------------------------------------- #


def test_frame_payload_is_readable_parquet():
    """Kept a valid (geo)parquet file so existing readers can consume cache objects."""
    import pyarrow.parquet as pq

    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    payload = FRAME.encode(gdf)
    assert payload[:4] == b"PAR1"
    assert b"geo" in (pq.read_schema(pa.BufferReader(payload)).metadata or {})


def test_keyed_payload_embeds_frames_raw_rather_than_base64():
    df = pd.DataFrame({"a": list(range(100))})
    value = [((("x", "=", "1"),), df)]
    payload = KEYED_ANY.encode(value)
    frame_payload = FRAME.encode(df)
    assert frame_payload in payload
    header = payload[: payload.index(frame_payload)]
    assert b"PAR1" not in header, "the frame must not also be inlined in the header"
    assert json.loads(header[header.index(b"[") :].rstrip(b"\n"))[0][1]["v"]["c"] == FRAME_CODEC_ID


def test_roundtrip_preserves_dicts_with_different_keys_per_row():
    """Arrow types a dict column as a struct over the *union* of its keys.

    So an ER event-details column, where no two events carry the same keys,
    comes back with every absent key filled in as None unless the column is
    stored as structure rather than as a type.
    """
    df = pd.DataFrame({"details": [{"k": 1}, {"j": "x"}, {}]})
    out = roundtrip(FRAME, df)
    assert out["details"].tolist() == [{"k": 1}, {"j": "x"}, {}]
    pd.testing.assert_frame_equal(out, df)


def test_roundtrip_preserves_nesting_inside_a_cell():
    df = pd.DataFrame({"details": [{"colour": (1, 2, 3), "tags": ["a"], "n": None}]})
    out = roundtrip(FRAME, df)
    assert out["details"][0] == {"colour": (1, 2, 3), "tags": ["a"], "n": None}


def test_a_cell_json_cannot_carry_leaves_the_frame_uncached():
    """Refused at encode, where wt-task turns it into "not cached"."""
    df = pd.DataFrame({"details": [{"when": object()}]})
    with pytest.raises(Undigestible):
        FRAME.encode(df)


def test_arrow_roundtrip_is_not_decided_by_geoarrow_import_order():
    """An extension *name* in field metadata must not become an extension *type*.

    Arrow promotes the one into the other on read when the extension is
    registered, and `geoarrow.pyarrow` registers on import -- so without this
    the same bytes decode to `binary` or to `extension<geoarrow.wkb>` depending
    on what else the process imported.
    """
    import geoarrow.pyarrow  # noqa: F401  # registers geoarrow.* process-wide

    table = arrow_table()
    out = roundtrip(ARROW, table)
    assert out.schema.field("geometry").type == pa.binary()
    assert out.schema.field("geometry").metadata == {b"ARROW:extension:name": b"geoarrow.wkb"}
    assert out.equals(table)


def test_arrow_roundtrip_of_a_real_geoarrow_extension_type():
    """What the warehouse client actually hands over: a genuine extension type."""
    import geoarrow.pyarrow as ga
    from shapely.geometry import Point as ShapelyPoint

    table = pa.table({"geometry": ga.array([ShapelyPoint(0, 0).wkb, ShapelyPoint(1, 1).wkb])})
    out = roundtrip(ARROW, table)
    assert out.schema.field("geometry").type == table.schema.field("geometry").type
    assert out.equals(table)
    assert ARROW.digest(out) == ARROW.digest(table)
