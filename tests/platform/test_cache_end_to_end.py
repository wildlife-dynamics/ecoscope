"""The headline invariant: a cache hit is indistinguishable from a miss.

Runs a chain of real registered tasks through `wt_task.cache.cached_call` twice
over a real store, which is the only way to catch a codec that round-trips in
isolation but not through the runtime -- and the only way to see uncacheability
propagate, since a step whose upstream could not be digested is itself
uncacheable however good its own codec is.
"""

import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point
from wt_task.cache import CacheMismatchWarning, CacheSpec, cached_call, run_scope

from ecoscope.platform.indexes import ValueGrouper
from ecoscope.platform.tasks.groupby import merge_df, set_groupers, split_groups
from ecoscope.platform.tasks.transformation._sorting import sort_values


@pytest.fixture
def events():
    return gpd.GeoDataFrame(
        {
            "event_type": ["fire", "poaching", "fire"],
            "time": pd.to_datetime(["2024-01-03T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]),
            "details": [{"n": 1}, {"m": "x"}, {}],
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("WT_CACHE_VERIFY", raising=False)


def spec(task_instance_id, anchor, function, deps=None):
    return CacheSpec(
        task_instance_id=task_instance_id,
        anchor=anchor,
        function=function,
        recipe=f"recipe-{function}",
        deps=deps or {},
    )


def assert_same(a, b):
    """Compare two step returns, including the keyed iterables `split_groups` emits."""
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(b, a)
        assert getattr(b, "crs", None) == getattr(a, "crs", None)
    elif isinstance(a, list):
        assert len(b) == len(a)
        for left, right in zip(a, b):
            if isinstance(left, tuple):
                assert left[0] == right[0]
                assert_same(left[1], right[1])
            else:
                assert_same(left, right)
    else:
        assert b == a


def execute(events, params=None):
    """Run the chain once, returning each step's value, hit flag and digest."""
    values, hits = {}, {}
    steps = [
        (
            # Params rather than a closure: a source task's inputs reach its key
            # only if they are declared, exactly as the compiler emits them.
            CacheSpec(
                task_instance_id="events",
                anchor="ecoscope.platform.tasks.io._earthranger",
                function="get_events",
                recipe="recipe-get_events",
                params=params or {"event_types": ["fire", "poaching"]},
            ),
            lambda: events.copy(),
        ),
        (
            spec("sorted", "ecoscope.platform.tasks.transformation._sorting", "sort_values", {"df": ["events"]}),
            lambda: sort_values(values["events"], column_name="time"),
        ),
        (
            spec("groupers", "ecoscope.platform.tasks.groupby._groupby", "set_groupers"),
            lambda: set_groupers([ValueGrouper(index_name="event_type")]),
        ),
        (
            spec(
                "split",
                "ecoscope.platform.tasks.groupby._groupby",
                "split_groups",
                {"df": ["sorted"], "groupers": ["groupers"]},
            ),
            lambda: split_groups(values["sorted"], values["groupers"]),
        ),
        (
            spec("merged", "ecoscope.platform.tasks.groupby._groupby", "merge_df", {"iterables": ["split"]}),
            lambda: merge_df(values["split"]),
        ),
    ]
    with run_scope() as ids:
        for cache_spec, run in steps:
            values[cache_spec.task_instance_id], hits[cache_spec.task_instance_id] = cached_call(cache_spec, run)
        digests = dict(ids)
    return values, hits, digests


def test_a_hit_is_indistinguishable_from_a_miss(cache_env, events):
    cold_values, cold_hits, cold_digests = execute(events)
    warm_values, warm_hits, warm_digests = execute(events)

    assert not any(cold_hits.values()), "the first run must execute everything"
    assert cold_digests == warm_digests

    for step, cold in cold_values.items():
        assert_same(cold, warm_values[step])


def test_the_frame_steps_hit_on_the_second_run(cache_env, events):
    execute(events)
    _, warm_hits, digests = execute(events)

    assert warm_hits["events"] and warm_hits["sorted"]
    assert digests["events"] is not None


def test_uncacheability_propagates_from_set_groupers(cache_env, events):
    """`set_groupers` returns `list[ValueGrouper | ...]`, which no codec claims.

    So the grouped subtree below it cannot cache, whatever the keyed-iterable
    codec does -- a key that silently omitted an unknown input would collide
    across genuinely different runs. Pinned because the fix is a grouper codec,
    and this test is what will tell you it worked.
    """
    _, _, digests = execute(events)

    assert digests["groupers"] is None
    assert digests["split"] is None
    assert digests["merged"] is None


def test_a_changed_param_misses_but_cuts_off_early(cache_env, events):
    """A step whose key moved re-runs; its dependents still hit if its output didn't.

    This is what promoting every step's output digest buys over keying dependents
    on recipe identity, and it fails loudly if that promotion regresses.
    """
    execute(events)
    _, hits, _ = execute(events, params={"event_types": ["wildlife"]})

    assert not hits["events"], "a changed param must move the key"
    assert hits["sorted"], "unchanged output must stop the recomputation there"


def test_changed_content_reaches_dependents(cache_env, events):
    execute(events)
    changed = events.copy()
    changed.loc[0, "event_type"] = "wildlife"
    _, hits, _ = execute(changed, params={"event_types": ["wildlife"]})

    assert not hits["events"]
    assert not hits["sorted"], "a changed upstream digest must reach its dependents"


def test_verify_mode_reports_nothing_for_a_deterministic_chain(cache_env, events, monkeypatch):
    execute(events)
    monkeypatch.setenv("WT_CACHE_VERIFY", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        execute(events)

    assert [w for w in caught if issubclass(w.category, CacheMismatchWarning)] == []


def test_verify_mode_names_a_task_that_does_not_reproduce(cache_env, events, monkeypatch):
    """What §7's survey relies on, and what a misspelled `no-cache` tag cannot evade."""
    execute(events)
    monkeypatch.setenv("WT_CACHE_VERIFY", "1")
    wobbled = events.copy()
    wobbled.loc[0, "details"] = [{"n": 2}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        execute(wobbled)

    mismatches = [str(w.message) for w in caught if issubclass(w.category, CacheMismatchWarning)]
    assert any("'events'" in m for m in mismatches), mismatches


def test_the_cache_is_inert_when_wt_cache_is_unset(events, monkeypatch):
    monkeypatch.delenv("WT_CACHE", raising=False)
    values, hits, digests = execute(events)

    assert not any(hits.values())
    assert set(digests.values()) == {None}
    pd.testing.assert_frame_equal(
        values["merged"],
        merge_df(split_groups(sort_values(events, "time"), set_groupers([ValueGrouper(index_name="event_type")]))),
    )
