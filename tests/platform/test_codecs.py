"""Conformance suite for the leaf codecs `ecoscope.platform` declares.

The headline property is `hit == miss`: whatever a codec stores must come back
indistinguishable from what went in, and the same value must digest the same
however it was encoded. Everything here is one of those two halves.

Which codec applies is no longer decided by inspecting a value, so the suite
starts from the declarations: if an alias or a class stopped declaring one,
every round-trip below would still pass while nothing cached.
"""

import datetime
import enum
import importlib
import json
import os
import pkgutil
import zoneinfo
from typing import Any
from unittest import mock

import dateutil.tz
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import pytz
from pandas.testing import assert_frame_equal
from pydantic import BaseModel
from shapely.geometry import LineString, Point
from wt_contracts import Cacheable, ModelDumpCacheable, declared_structure
from wt_task.cache import Tree, Undigestible, resolve_structure
from wt_task.skip import SKIP_SENTINEL

import ecoscope.platform
from ecoscope.platform import codecs
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.codecs import (
    ARROW,
    ARROW_CODEC_ID,
    FRAME,
    FRAME_CODEC_ID,
    KEYED_ANY,
    UnencodableValue,
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


TIMEZONES = [
    None,
    datetime.timezone.utc,
    datetime.timezone(datetime.timedelta(hours=-5, minutes=-30), "Somewhere"),
    zoneinfo.ZoneInfo("Africa/Nairobi"),
    pytz.timezone("Europe/London"),
    pytz.FixedOffset(90),
]


def assert_same_cell(out, expected):
    """Equal, of the same type, and printing alike -- the digest of a mixed column is over `repr`."""
    assert type(out) is type(expected)
    assert repr(out) == repr(expected)
    if expected is not pd.NaT:
        assert out == expected


@pytest.mark.parametrize("tz", TIMEZONES, ids=repr)
def test_roundtrip_preserves_timestamps_nested_in_cells(tz):
    """ER patrol segments carry parsed times inside their dict and list cells."""
    start = pd.Timestamp("2024-03-31T00:30:00.123456789", tz=tz)
    df = pd.DataFrame({"segments": [{"time_range": [start, pd.NaT]}, {"time_range": (start, None)}]})
    out = roundtrip(FRAME, df)
    assert_same_cell(out["segments"][0]["time_range"][0], start)
    assert out["segments"][0]["time_range"][1] is pd.NaT
    assert out["segments"][1] == df["segments"][1]
    assert digest_dataframe(out) == digest_dataframe(df)


@pytest.mark.parametrize("tz", TIMEZONES, ids=repr)
def test_roundtrip_preserves_timestamps_mixed_into_an_object_column(tz):
    in_the_fold = pd.Timestamp("2024-10-27T00:30:00")  # 01:30 in London, for the second time
    cells = [in_the_fold if tz is None else in_the_fold.tz_localize("UTC").tz_convert(tz), "unknown", pd.NaT, None, 3]
    df = pd.DataFrame({"t": pd.Series(cells, dtype=object)})
    out = roundtrip(FRAME, df)
    assert out["t"].dtype == object
    for got, expected in zip(out["t"], cells):
        assert_same_cell(got, expected)
    assert digest_dataframe(out) == digest_dataframe(df)


def test_roundtrip_keeps_an_object_column_of_timestamps_as_objects():
    """Arrow would retype it as one datetime column, in one zone."""
    cells = [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz=zoneinfo.ZoneInfo("Africa/Nairobi"))]
    df = pd.DataFrame({"t": pd.Series(cells, dtype=object)})
    out = roundtrip(FRAME, df)
    assert out["t"].dtype == object
    for got, expected in zip(out["t"], cells):
        assert_same_cell(got, expected)
    assert digest_dataframe(out) == digest_dataframe(df)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_roundtrip_preserves_a_timestamp_cells_unit(unit):
    df = pd.DataFrame({"t": [[pd.Timestamp("2024-01-01", tz="UTC").as_unit(unit)]]})
    assert roundtrip(FRAME, df)["t"][0][0].unit == unit


@pytest.mark.parametrize("tz", TIMEZONES, ids=repr)
def test_roundtrip_preserves_datetime_cells(tz):
    moment = datetime.datetime(2024, 10, 27, 1, 30, 0, 123456)
    moment = moment if tz is None else moment.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
    df = pd.DataFrame({"t": [{"at": moment}, {"at": None}]})
    out = roundtrip(FRAME, df)
    assert_same_cell(out["t"][0]["at"], moment)
    assert digest_dataframe(out) == digest_dataframe(df)


@pytest.mark.parametrize(
    "delta",
    [pd.Timedelta(-1500, "ms"), pd.Timedelta(1, "ns"), pd.Timedelta(400, "D").as_unit("s")],
    ids=str,
)
def test_roundtrip_preserves_timedelta_cells(delta):
    df = pd.DataFrame({"d": [[delta], [pd.NaT]]})
    out = roundtrip(FRAME, df)
    assert_same_cell(out["d"][0][0], delta)
    assert out["d"][0][0].unit == delta.unit
    assert out["d"][1][0] is pd.NaT
    assert digest_dataframe(out) == digest_dataframe(df)


@pytest.mark.parametrize(
    "tz",
    [datetime.timezone(datetime.timedelta(0), "UTC"), datetime.timezone(datetime.timedelta(hours=3), "UTC+03:00")],
    ids=repr,
)
def test_roundtrip_keeps_a_fixed_offset_name_that_matches_the_default(tz):
    """`timezone(0, "UTC")` is what `astimezone()` gives on a UTC host, and it does not print like `timezone.utc`."""
    moment = datetime.datetime(2024, 1, 1, 12, tzinfo=tz)
    df = pd.DataFrame({"t": [{"at": moment}]})
    out = roundtrip(FRAME, df)
    assert_same_cell(out["t"][0]["at"], moment)
    assert digest_dataframe(out) == digest_dataframe(df)


@pytest.mark.parametrize(
    "moment",
    [
        datetime.datetime(2024, 1, 1, 12).replace(tzinfo=pytz.timezone("Africa/Nairobi")),  # LMT+2:27
        pytz.timezone("Europe/London").localize(datetime.datetime(2024, 10, 27, 1, 30), is_dst=True),
        datetime.datetime(2024, 3, 31, 1, 30, tzinfo=zoneinfo.ZoneInfo("Europe/London")),  # in the gap
        datetime.datetime(2024, 10, 27, 1, 30, fold=1, tzinfo=zoneinfo.ZoneInfo("Europe/London")),
        datetime.datetime(2024, 10, 27, 1, 30, fold=1),
    ],
    ids=repr,
)
def test_roundtrip_keeps_a_datetime_cells_wall_clock(moment):
    df = pd.DataFrame({"t": [{"at": moment}]})
    out = roundtrip(FRAME, df)
    assert_same_cell(out["t"][0]["at"], moment)
    assert digest_dataframe(out) == digest_dataframe(df)


@pytest.mark.parametrize(
    "stamp",
    [
        pd.Timestamp(np.datetime64("-0100-01-01", "s")),
        pd.Timestamp(np.datetime64("20000-01-01", "s")),
        pd.Timestamp(year=2024, month=10, day=27, hour=1, minute=30, fold=1),
    ],
    ids=repr,
)
def test_roundtrip_preserves_timestamps_text_cannot_carry(stamp):
    df = pd.DataFrame({"t": [[stamp]]})
    out = roundtrip(FRAME, df)["t"][0][0]
    assert_same_cell(out, stamp)
    assert (out.unit, out.fold) == (stamp.unit, stamp.fold)


def test_roundtrip_preserves_a_timedelta_cell_beyond_the_nanosecond_range():
    delta = pd.Timedelta.max.as_unit("s") * 5
    out = roundtrip(FRAME, pd.DataFrame({"d": [[delta]]}))["d"][0][0]
    assert_same_cell(out, delta)
    assert out.unit == "s"


def test_roundtrip_preserves_stdlib_timedelta_cells():
    """Arrow would retype a column of them as `timedelta64`."""
    cells = [datetime.timedelta(seconds=5), datetime.timedelta(days=-1, microseconds=7)]
    df = pd.DataFrame({"d": pd.Series(cells, dtype=object), "nested": [[cells[0], pd.Timedelta(1, "s")], [cells[1]]]})
    out = roundtrip(FRAME, df)
    assert out["d"].dtype == object
    for got, expected in zip(out["d"], cells):
        assert_same_cell(got, expected)
    assert_same_cell(out["nested"][0][0], cells[0])
    assert digest_dataframe(out) == digest_dataframe(df)


def _object_frame(cells):
    return pd.DataFrame({"t": pd.Series(cells, dtype=object)})


@pytest.mark.parametrize(
    "other",
    [
        "2024-01-01 03:00:00+03:00",
        pd.Timestamp("2024-01-01T03:00", tz=datetime.timezone(datetime.timedelta(hours=3))),
        pd.Timestamp("2024-01-01T03:00", tz=pytz.timezone("Africa/Nairobi")),
        pd.Timestamp("2024-01-01T03:00", tz=zoneinfo.ZoneInfo("Africa/Nairobi")).as_unit("ns"),
        datetime.datetime(2024, 1, 1, 3, tzinfo=zoneinfo.ZoneInfo("Africa/Nairobi")),
    ],
    ids=repr,
)
def test_an_object_column_of_timestamps_digests_by_type_zone_and_unit(other):
    """pandas hashes such a column through `str`, under which all of these are one value."""
    stamp = pd.Timestamp("2024-01-01T03:00", tz=zoneinfo.ZoneInfo("Africa/Nairobi"))
    assert str(stamp) == str(other)
    assert digest_dataframe(_object_frame([stamp])) != digest_dataframe(_object_frame([other]))


def test_a_cell_of_an_unknown_kind_is_refused_rather_than_read_as_a_dict():
    with pytest.raises(ValueError, match="unreadable cell node"):
        codecs._cell_from_json({"k": "from-the-future", "v": "x"})


def test_a_timestamp_in_a_zone_the_codec_cannot_name_is_declined():
    """Stored under another zone object, a hit would digest differently from a miss."""
    df = pd.DataFrame({"t": [[pd.Timestamp("2024-01-01", tz=dateutil.tz.tzutc())]]})
    with pytest.raises(UnencodableValue, match="timezone"):
        FRAME.encode(df)


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


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": pd.Series([], dtype="int64")}),
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [1, 2, 3]}),
        pd.DataFrame({"a": [1, 2]}, index=pd.RangeIndex(5, 9, 2, name="r")),
        gpd.GeoDataFrame({"a": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"),
        pd.DataFrame({"a": [[1], (2,)]}),
    ],
    ids=["empty", "one-row", "several-rows", "offset-step-named", "geo", "tree-column"],
)
def test_roundtrip_keeps_a_range_index(df):
    """Tasks branch on `RangeIndex`, so an int64 `Index` of the same values is a different hit."""
    out = roundtrip(FRAME, df)
    pd.testing.assert_index_equal(out.index, df.index, exact=True)


def test_roundtrip_keeps_an_int_index_that_is_not_a_range():
    df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([0, 1]))
    out = roundtrip(FRAME, df)
    pd.testing.assert_index_equal(out.index, df.index, exact=True)


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
# frame round-trips -- strict, across the shapes parquet does not carry as is
# --------------------------------------------------------------------------- #

_P = [Point(0, 0), Point(1, 1), Point(2, 2)]


def _with_attrs(df, attrs):
    df.attrs = attrs
    return df


STRICT_FRAMES = {
    # indexes
    "multiindex": pd.DataFrame({"x": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["g", "v"])),
    "multiindex with int names": pd.DataFrame(
        {"x": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=[0, 1])
    ),
    "multiindex with categorical level": pd.DataFrame(
        {"x": [1, 2]}, index=pd.MultiIndex.from_arrays([pd.Categorical([1, 2]), ["a", "b"]], names=["c", "s"])
    ),
    "multiindex from groupby": pd.DataFrame({"g": ["a", "a", "b"], "h": [1, 2, 1], "x": [1, 2, 3]})
    .groupby(["g", "h"])
    .sum(),
    "range index, negative step": pd.DataFrame({"x": [1, 2]}, index=pd.RangeIndex(3, 1, -1)),
    "int index with gaps": pd.DataFrame({"x": [1, 2, 3]}).iloc[[0, 2]],
    "string-dtype index": pd.DataFrame({"x": [1, 2]}, index=pd.Index(["a", "b"], dtype="string")),
    "index named 0": pd.DataFrame({"x": [1, 2]}, index=pd.Index(["p", "q"], name=0)),
    "datetime index with freq": pd.DataFrame(
        {"x": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, tz="Africa/Nairobi")
    ),
    "timedelta index with freq": pd.DataFrame({"x": [1, 2]}, index=pd.timedelta_range("1h", periods=2, freq="h")),
    "categorical index, int categories": pd.DataFrame({"x": [1, 2]}, index=pd.CategoricalIndex([1, 2], name="k")),
    "object index mixing types": pd.DataFrame({"x": [1, 2]}, index=pd.Index([1, "a"], dtype=object)),
    "object index with NaN": pd.DataFrame({"x": [1, 2]}, index=pd.Index(["a", np.nan], dtype=object)),
    # column labels
    "range columns": pd.DataFrame([[1, 2]]),
    "no columns": pd.DataFrame(),
    "duplicate labels": pd.DataFrame([[1, 2]], columns=["a", "a"]),
    "bool labels": pd.DataFrame([[1, 2]], columns=[True, False]),
    "mixed labels": pd.DataFrame([[1, 2]], columns=pd.Index([0, "a"], dtype=object)),
    "tuple labels": pd.DataFrame([[1, 2]], columns=pd.Index([("a", 1), ("b", 2)], tupleize_cols=False)),
    "multiindex labels with ints": pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])),
    "duplicate multiindex labels": pd.DataFrame(
        [[1, 2]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "x")], names=["l", "m"])
    ),
    # dtypes
    "datetime64[s]": pd.DataFrame({"a": pd.to_datetime(["2024-01-01", None]).astype("datetime64[s]")}),
    "datetime, zoneinfo zone": pd.DataFrame(
        {"a": pd.date_range("2024-01-01", periods=2, tz=zoneinfo.ZoneInfo("Africa/Nairobi"))}
    ),
    "timedelta64[s]": pd.DataFrame({"a": pd.to_timedelta([1, None], unit="h").astype("timedelta64[s]")}),
    "string[pyarrow]": pd.DataFrame({"a": pd.Series(["x", None], dtype="string[pyarrow]")}),
    "sparse": pd.DataFrame({"a": pd.arrays.SparseArray([0, 0, 1])}),
    "sparse complex": pd.DataFrame({"a": pd.arrays.SparseArray([0j, 0j, 1 + 1j])}),
    "complex": pd.DataFrame({"a": np.array([1 + 2j, np.nan], dtype="complex128")}),
    "categorical, int categories": pd.DataFrame({"a": pd.Categorical([1, 2, 1])}),
    "categorical, ordered, unused category": pd.DataFrame(
        {"a": pd.Categorical([3, 1], categories=[3, 2, 1], ordered=True)}
    ),
    "categorical, datetime categories": pd.DataFrame(
        {"a": pd.Categorical(pd.to_datetime(["2024-01-01", "2024-01-02"]))}
    ),
    "categorical from pd.cut": pd.DataFrame({"a": pd.cut([1, 5, 9], bins=[0, 3, 6, 10])}),
    # object columns
    "object ints": pd.DataFrame({"a": pd.Series([1, 2], dtype=object)}),
    "object ints with None": pd.DataFrame({"a": pd.Series([1, None], dtype=object)}),
    "object bools": pd.DataFrame({"a": pd.Series([True, False], dtype=object)}),
    "object numpy scalars": pd.DataFrame({"a": pd.Series([np.int64(1), np.float32(1.5), np.str_("x")], dtype=object)}),
    "strings with NaN (left merge)": pd.DataFrame({"k": [1, 2]}).merge(
        pd.DataFrame({"k": [1], "s": ["x"]}), how="left"
    ),
    "strings with pd.NA": pd.DataFrame({"a": pd.Series(["x", pd.NA], dtype=object)}),
    "NaN beside None": pd.DataFrame({"a": ["x", np.nan, None]}),
    "all NaN": pd.DataFrame({"a": pd.Series([np.nan, np.nan], dtype=object)}),
    "decimals": pd.DataFrame({"a": [__import__("decimal").Decimal("1.10"), None]}),
    "bytes, dates, times mixed": pd.DataFrame({"a": [b"x", datetime.date(2024, 1, 1), datetime.time(1, 2)]}),
    "sets, frozensets, int-keyed dicts": pd.DataFrame({"a": [{1, 2}, frozenset({3}), {1: "x"}]}),
    "numpy array cells": pd.DataFrame({"a": [np.array([1, 2]), np.array([[1.5]]), None]}),
    "NaN inside a list": pd.DataFrame({"a": [[1.0, np.nan], [None]]}),
    "uuids": pd.DataFrame({"a": [__import__("uuid").UUID(int=1), None]}),
    "shapely in an object column": pd.DataFrame({"a": [Point(0, 0), None]}),
    "numpy datetime and timedelta cells": pd.DataFrame(
        {
            "a": pd.Series([np.timedelta64(5, "s"), np.timedelta64("NaT", "s"), np.datetime64(1, "D")], dtype=object),
            "b": [[np.timedelta64(5, "s")], [np.timedelta64(3)], []],
        }
    ),
    # frames
    "flags": pd.DataFrame({"a": [1]}).set_flags(allows_duplicate_labels=False),
    "nested attrs": _with_attrs(pd.DataFrame({"a": [1]}), {"nested": {"k": [1, 2]}}),
    "geo, tree column": gpd.GeoDataFrame({"d": [{"k": 1}, [1], (2,)]}, geometry=_P, crs=4326),
    "geo, string-NaN column": gpd.GeoDataFrame({"s": ["x", np.nan, "z"]}, geometry=_P, crs=4326),
    "geo, categorical index": gpd.GeoDataFrame(
        {"a": [1, 2, 3]}, geometry=_P, crs=4326, index=pd.CategoricalIndex([1.5, 2.5, 1.5])
    ),
    "geo without geometry": gpd.GeoDataFrame({"a": [1, 2]}),
    "geo, no active geometry": gpd.GeoDataFrame({"a": [1, 2, 3], "g": gpd.GeoSeries(_P, crs=4326)}),
    "geo, range labels": gpd.GeoDataFrame(pd.DataFrame([[1, p] for p in _P]), geometry=1, crs=4326),
    "geo, attrs": _with_attrs(
        gpd.GeoDataFrame({"a": [1, 2, 3]}, geometry=_P, crs=4326), {"grouper": {"index_name": "month"}}
    ),
    "geo, int labels": gpd.GeoDataFrame({0: [1, 2, 3], "pt": _P}, geometry="pt", crs=4326),
    "geo, duplicate labels": gpd.GeoDataFrame(
        pd.DataFrame([[1, 2, _P[0]]], columns=["a", "a", "geometry"]), geometry="geometry", crs=4326
    ),
    "attrs JSON cannot hold": _with_attrs(pd.DataFrame({"a": [1]}), {"since": datetime.date(2024, 1, 1), 1: (2, 3)}),
    "plain frame holding geometry": pd.DataFrame({"g": gpd.GeoSeries(_P, crs=4326)}),
    "plain frame holding geometry, range labels": pd.DataFrame({"g": gpd.GeoSeries(_P, crs=4326)}).set_axis(
        pd.RangeIndex(1), axis=1
    ),
    "Int64 index with a missing value": pd.DataFrame({"x": [1, 2]}, index=pd.Index([1, None], dtype="Int64")),
    "boolean index": pd.DataFrame({"x": [1, 2]}, index=pd.Index([True, False], dtype="boolean")),
    "Float64 index": pd.DataFrame({"x": [1, 2]}, index=pd.Index([1.5, None], dtype="Float64")),
    "multiindex with an Int64 level": pd.DataFrame(
        {"x": [1]}, index=pd.MultiIndex.from_arrays([pd.array([1], dtype="Int64"), ["a"]])
    ),
    "plain frame holding geometry, attrs": _with_attrs(pd.DataFrame({"g": gpd.GeoSeries(_P, crs=4326)}), {"k": 1}),
}


def _cells_by_repr(df):
    """Object columns as the `repr` of each cell, which names its type and lets NaN equal NaN."""
    out = df.copy()
    for i, dtype in enumerate(df.dtypes):
        if dtype == object:
            out.isetitem(i, [repr(v) for v in df.iloc[:, i]])
    return out


@pytest.mark.parametrize("df", STRICT_FRAMES.values(), ids=STRICT_FRAMES.keys())
def test_roundtrip_is_exact(df):
    out = roundtrip(FRAME, df)
    assert type(out) is type(df)
    pd.testing.assert_frame_equal(
        _cells_by_repr(out),
        _cells_by_repr(df),
        check_index_type=True,
        check_column_type=True,
        check_exact=True,
        check_freq=True,
        check_flags=True,
    )
    assert out.attrs == df.attrs
    if isinstance(df, gpd.GeoDataFrame):
        assert out._geometry_column_name == df._geometry_column_name
    assert digest_dataframe(out) == digest_dataframe(df)


class _Str(enum.StrEnum):
    A = "a"


class _Int(enum.IntEnum):
    X = 1


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [datetime.time(1, tzinfo=datetime.timezone.utc)]}),
        pd.DataFrame({"a": pd.Series([_Str.A], dtype=object)}),
        pd.DataFrame({"a": [[_Str.A]]}),
        pd.DataFrame({"a": [{_Int.X: 1}]}),
        pd.DataFrame({_Str.A: [1]}),
    ],
    ids=["tz-aware time", "enum cell", "enum in a list", "enum dict key", "enum label"],
)
def test_what_would_read_back_as_something_else_is_declined_outside_strict_too(df, monkeypatch):
    monkeypatch.setenv("WT_CACHE__STRICT", "false")
    with pytest.raises(Undigestible):
        FRAME.encode(df)


def test_a_set_cell_digests_alike_in_every_process():
    """Iteration order follows `PYTHONHASHSEED`, so a digest of it would never hit in another process."""
    import subprocess
    import sys

    script = (
        "import pandas as pd; from ecoscope.platform.codecs import digest_dataframe; "
        "print(digest_dataframe(pd.DataFrame({'a': [{'apple', 'banana', 'cherry'}, {'d': frozenset('xyz')}]})))"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(digests) == 1


def test_array_cells_differing_past_reprs_truncation_digest_apart():
    a = np.zeros(2000)
    b = a.copy()
    b[1000] = 1
    assert digest_dataframe(pd.DataFrame({"a": [a]})) != digest_dataframe(pd.DataFrame({"a": [b]}))
    assert digest_dataframe(pd.DataFrame({"a": [[a]]})) != digest_dataframe(pd.DataFrame({"a": [[b]]}))


def test_geometry_cells_differing_past_reprs_truncation_digest_apart():
    a = LineString([(i, i) for i in range(30)])
    b = LineString([(i, i) for i in range(29)] + [(99, 99)])
    assert digest_dataframe(pd.DataFrame({"a": [a]})) != digest_dataframe(pd.DataFrame({"a": [b]}))
    assert digest_dataframe(pd.DataFrame({"a": [[a]]})) != digest_dataframe(pd.DataFrame({"a": [[b]]}))


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": pd.Series([[1], None], dtype=pd.ArrowDtype(pa.list_(pa.int64())))}),
        pd.DataFrame({"a": [datetime.time(1, tzinfo=datetime.timezone.utc)]}),
    ],
    ids=["nested arrow dtype", "tz-aware time cell"],
)
def test_a_frame_that_would_not_read_back_is_declined(df, monkeypatch):
    """Left uncached rather than served as something other than what the step returned."""
    monkeypatch.setenv("WT_CACHE__STRICT", "true")
    with pytest.raises(Undigestible):
        FRAME.encode(df)


def test_the_read_back_check_is_skipped_outside_strict(monkeypatch):
    """A deployed run does not pay for the extra decode and digests."""
    monkeypatch.setenv("WT_CACHE__STRICT", "false")
    decode = mock.Mock(wraps=codecs.decode_dataframe)
    monkeypatch.setattr(codecs, "decode_dataframe", decode)
    FRAME.encode(pd.DataFrame({"a": [1]}))
    decode.assert_not_called()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (pd.DataFrame({"a": ["x", np.nan]}), pd.DataFrame({"a": ["x", None]})),
        (pd.DataFrame({"a": pd.Series([1, 2], dtype=object)}), pd.DataFrame({"a": ["1", "2"]})),
        (pd.DataFrame({"a": pd.Categorical([1, 2])}), pd.DataFrame({"a": pd.Categorical([1, 2], ordered=True)})),
        (
            pd.DataFrame({"a": pd.Series(["x"], dtype="string[python]")}),
            pd.DataFrame({"a": pd.Series(["x"], dtype="string[pyarrow]")}),
        ),
        (
            pd.DataFrame({"x": [1]}, index=pd.Index(["p"], name=0)),
            pd.DataFrame({"x": [1]}, index=pd.Index(["p"], name="0")),
        ),
        (
            pd.DataFrame({"x": [1, 2]}, index=pd.Index([1, "a"])),
            pd.DataFrame({"x": [1, 2]}, index=pd.Index(["1", "a"])),
        ),
        (pd.DataFrame([[1]]), pd.DataFrame([[1]], columns=pd.Index([0]))),
        (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}).set_flags(allows_duplicate_labels=False)),
        (
            pd.DataFrame({"x": [1, 2]}, index=pd.date_range("2024-01-01", periods=2)),
            pd.DataFrame({"x": [1, 2]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"])),
        ),
    ],
    ids=[
        "NaN vs None",
        "object ints vs strings",
        "ordered",
        "string storage",
        "int vs str index name",
        "object index mixed vs strings",
        "range vs int columns",
        "flags",
        "index freq",
    ],
)
def test_digest_distinguishes_what_the_codec_preserves(a, b):
    """Otherwise a hit could stand in for a miss that tasks treat differently."""
    assert digest_dataframe(a) != digest_dataframe(b)


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


def test_digest_distinguishes_a_range_index_from_an_int_index_of_the_same_values():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"x": [1, 2]}, index=pd.Index([0, 1]))
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
    """Refuses subclasses, so a payload can never be decoded back into the wrong class."""
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
    "DataSetManifest": {
        "data_set": "ds",
        "tables": {"events": {"path": "events.parquet", "rows": 1, "schema_version": "1"}},
    },
    "TileLayer": {"layer_name": "TERRAIN"},
    "CustomMetric": {"display_name": "Total", "aggregator": "sum", "column": "distance"},
    "TrajectorySegmentFilterOverride": {"key": "elephant", "filter": {"preset": "elephant", "max_speed_kmhr": 12}},
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
DECLARED_MODELS = [cls for cls in _declared_models().values() if issubclass(cls, ModelDumpCacheable)]
SELF_ENCODING_MODELS = [cls for cls in _declared_models().values() if not issubclass(cls, ModelDumpCacheable)]


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


def test_only_the_groupers_encode_themselves():
    """A model declared through plain `Cacheable` skips the structural check, so it must stay rare and deliberate.

    A model declared that way without a codec that really does encode it would
    be cached through `model_dump` with no check that the dump survives.
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
    """Refuses subclasses: encoding one through its parent would narrow it silently."""

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
