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
from wt_task.cache import CacheMismatchWarning, CacheSpec, cached_call, resolve_task, run_scope

from ecoscope.platform.indexes import ValueGrouper
from ecoscope.platform.tasks.groupby import groupbykey, merge_df, set_groupers, split_groups
from ecoscope.platform.tasks.io._earthranger import get_events
from ecoscope.platform.tasks.results._map_utils import persist_geoarrow_for_pydeck
from ecoscope.platform.tasks.results._pydeck import create_geoarrow_scatterplot_layer
from ecoscope.platform.tasks.transformation import align_keyed_iterable_to_reference
from ecoscope.platform.tasks.transformation._mapping import map_columns
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
    monkeypatch.setenv("WT_CACHE__STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("WT_CACHE__VERIFY", raising=False)


def spec(task, deps=None, **kwargs):
    """The half of a step's identity the compiler emits -- hashes of spec text.

    The task's own identity and the structure of its return are worked out at
    run time by `resolve_task`, so a declaration that stopped resolving shows
    up here as an uncached step rather than as a passing test.
    """
    return CacheSpec(signature=f"signature-{task.__name__}", upstream_inputs=deps or {}, **kwargs)


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


def execute(events, params=None, groupers=None):
    """Run the chain once, returning each step's value, hit flag and digest."""
    values, hits = {}, {}
    groupers = [ValueGrouper(index_name="event_type")] if groupers is None else groupers
    steps = [
        ("events", get_events, spec(get_events), lambda: events.copy()),
        (
            "sorted",
            sort_values,
            spec(sort_values, {"df": ["events"]}),
            lambda: sort_values(values["events"], column_name="time"),
        ),
        ("groupers", set_groupers, spec(set_groupers), lambda: set_groupers(groupers)),
        (
            "split",
            split_groups,
            spec(split_groups, {"df": ["sorted"], "groupers": ["groupers"]}),
            lambda: split_groups(values["sorted"], values["groupers"]),
        ),
        ("merged", merge_df, spec(merge_df, {"iterables": ["split"]}), lambda: merge_df(values["split"])),
    ]
    # Through the run's parameter table rather than a closure: a source task's
    # inputs reach its key only this way, exactly as the DAG passes them.
    run_params = {
        "events": params or {"event_types": ["fire", "poaching"]},
        "groupers": {"groupers": [g.index_name for g in groupers]},
    }
    with run_scope(params=run_params) as ids:
        for task_instance_id, task, cache_spec, run in steps:
            resolved = resolve_task(task)
            assert resolved.tree("call") is not None, f"{task_instance_id} declares no codec"
            values[task_instance_id], hits[task_instance_id] = cached_call(cache_spec, run, resolved, task_instance_id)
        # `run_scope` records an outcome alongside each digest; the digest is
        # what dependents key on, and what these tests are about.
        digests = {step: row.digest for step, row in ids.items()}
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


def test_the_grouped_subtree_caches(cache_env, events):
    """The grouper codec is what unblocks everything below `set_groupers`.

    Its output is an input to `split_groups`, so before it could be digested
    the whole grouped branch recorded itself uncacheable -- a key that silently
    omitted an unknown input would collide across genuinely different runs.
    """
    _, _, cold_digests = execute(events)
    _, warm_hits, warm_digests = execute(events)

    assert all(cold_digests[step] is not None for step in ("groupers", "split", "merged"))
    assert cold_digests == warm_digests
    assert all(warm_hits.values()), "every step must serve from cache on the second run"


def test_the_ungrouped_subtree_caches(cache_env, events):
    """A blank `groupers` is a bare `AllGrouper`, not an empty list.

    So the workflows with no groupers at all need the single-value codec, and
    would otherwise be the ones that cache least.
    """
    _, _, cold_digests = execute(events, groupers=[])
    _, warm_hits, warm_digests = execute(events, groupers=[])

    assert cold_digests["groupers"] is not None
    assert cold_digests == warm_digests
    assert all(warm_hits.values())


def test_a_changed_grouper_reaches_the_grouped_subtree(cache_env, events):
    execute(events)
    _, hits, _ = execute(events, groupers=[])

    assert hits["sorted"], "the frame branch is untouched by a grouper change"
    assert not hits["split"], "a changed grouper digest must reach its dependents"


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
    monkeypatch.setenv("WT_CACHE__VERIFY", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        execute(events)

    assert [w for w in caught if issubclass(w.category, CacheMismatchWarning)] == []


def test_verify_mode_names_a_task_that_does_not_reproduce(cache_env, events, monkeypatch):
    """What §7's survey relies on, and what a misspelled `no-cache` tag cannot evade."""
    execute(events)
    monkeypatch.setenv("WT_CACHE__VERIFY", "1")
    wobbled = events.copy()
    wobbled.loc[0, "details"] = [{"n": 2}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        execute(wobbled)

    mismatches = [str(w.message) for w in caught if issubclass(w.category, CacheMismatchWarning)]
    assert any("'events'" in m for m in mismatches), mismatches


def test_the_cache_is_inert_when_wt_cache_is_unset(events, monkeypatch):
    monkeypatch.delenv("WT_CACHE__STORE", raising=False)
    values, hits, digests = execute(events)

    assert not any(hits.values())
    assert set(digests.values()) == {None}
    pd.testing.assert_frame_equal(
        values["merged"],
        merge_df(split_groups(sort_values(events, "time"), set_groupers([ValueGrouper(index_name="event_type")]))),
    )


def test_a_mapped_step_hands_its_dependents_tuple_keys(cache_env, events):
    """The shape the events spec is built from, and the one that broke.

    A `mapvalues` step's keys are whatever `argvalues` carried, so the step's
    structure walks them at run time rather than declaring them. JSON has no
    tuple, so a key that round-tripped as a list came back unhashable --
    `groupbykey` and `align_keyed_iterable_to_reference` both use it as a dict
    key, which made a hit crash where a miss had not.
    """
    groups = split_groups(events, set_groupers([ValueGrouper(index_name="event_type")]))
    urls = [(key, f"/results/{key[0][2]}.parquet") for key, _ in groups]

    def mapped(task_instance_id, task, value):
        cache_spec = spec(task)
        resolved = resolve_task(task)
        assert resolved.tree("mapvalues") is not None, f"{task_instance_id} declares no codec"
        with run_scope():
            call = (cache_spec, lambda: value, resolved, task_instance_id, "mapvalues")
            cached_call(*call)  # cold
            served, hit = cached_call(*call)
        assert hit, f"{task_instance_id} did not serve from cache"
        return served

    warm_groups = mapped("renamed", map_columns, groups)
    warm_urls = mapped("persisted", persist_geoarrow_for_pydeck, urls)

    assert all(isinstance(key, tuple) for key, _ in warm_groups)
    assert [key for key, _ in warm_groups] == [key for key, _ in groups]

    # The consumers, which is where a list key actually bites.
    combined = groupbykey([warm_groups, warm_urls])
    assert [key for key, _ in combined] == [key for key, _ in groupbykey([groups, urls])]
    aligned = align_keyed_iterable_to_reference(warm_groups, warm_urls)
    assert [key for key, _ in aligned] == [key for key, _ in groups]


def test_a_persisted_path_does_not_move_its_dependents_across_run_roots(cache_env, events, tmp_path, monkeypatch):
    """Each run persists under a fresh root, which reaches the persist step's key
    through its env binding. The path it returns is relative to that root, so its
    digest -- and so every dependent's key -- is the same run to run."""

    def run(root):
        monkeypatch.setenv("RESULTS", str(tmp_path / root))
        values, hits = {}, {}
        steps = [
            (
                "persisted",
                persist_geoarrow_for_pydeck,
                spec(persist_geoarrow_for_pydeck, env=["RESULTS"], servable=False),
                lambda: persist_geoarrow_for_pydeck(events, str(tmp_path / root)),
            ),
            (
                "layer",
                create_geoarrow_scatterplot_layer,
                spec(create_geoarrow_scatterplot_layer, {"data_url": ["persisted"]}),
                lambda: create_geoarrow_scatterplot_layer(values["persisted"]),
            ),
        ]
        with run_scope():
            for task_instance_id, task, cache_spec, call in steps:
                resolved = resolve_task(task)
                assert resolved.tree("call") is not None, f"{task_instance_id} declares no codec"
                values[task_instance_id], hits[task_instance_id] = cached_call(
                    cache_spec, call, resolved, task_instance_id
                )
        return values, hits

    first, _ = run("run-1")
    second, hits = run("run-2")

    assert (tmp_path / "run-2" / second["persisted"]).is_file(), "the file must still be written into this run's root"
    assert second["persisted"] == first["persisted"]
    assert hits == {"persisted": False, "layer": True}
