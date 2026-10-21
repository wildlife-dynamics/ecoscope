# Patrol Leg Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ecoscope task library read the new EarthRanger patrol leg fields and offer leg-based metrics and team/member/asset groupers, without changing any existing metric at a site that lacks the `patrol_schemas` flag.

**Architecture:** Ingest carries the new leg fields onto patrol observations and builds a separate one-row-per-leg frame. Four new presets read that leg frame through a second `summarize_df` call whose result is joined onto the existing trajectory summary, so existing numbers cannot move. Groupers over list columns (members, assets) are exploded at run time only when selected; leg metrics under spatial groupers are apportioned by the share of each leg's track length in each feature.

**Tech Stack:** Python 3.12, pandas, geopandas, pydantic v2, Pandera, `wt_registry.register` task decorator, pytest. Tests run from the worktree with `.venv/bin/python -m pytest` (created by `uv venv .venv && uv pip install -p .venv/bin/python -e ".[analysis,mapping,plotting,platform]" --group test`).

**Spec:** `docs/superpowers/specs/2026-10-20-patrol-leg-metrics-design.md` (canonical copy: https://claude.ai/artifact/CaEvhS5ZPwJbHApTCXde5F). Also read the Patrol Schema Change doc at https://claude.ai/artifact/89ZzTAHie8zgGDjfEEPt87 for API field semantics.

## Global Constraints

- Branch from upstream `master` (v2.25.1). Never from `dashbuilder-poc`.
- No existing preset is renamed, removed, or has its formula or column changed. `set_patrol_summary_metrics()` with no arguments must keep returning exactly today's six display names in today's order.
- Every new column is added only when its source field is present in the API response; a flag-off site produces no `patrol_team`, `patrol_members`, `patrol_assets`, `patrol_segment_details`, `patrol_type_details` column. `patrol_segment_id` is always added (segment `id` exists on every site).
- Pause legs (`is_pause: true`) are skipped before observations are fetched and excluded from the leg frame. `include_pauses` is never sent.
- Name columns `patrol_team_name`, `patrol_member_names`, `patrol_asset_names` always exist after their task runs; the fill for an absent source column is the string `"None"`.
- Durations: seconds internally; presets convert to `h` or `d` with `decimal_places=2`, matching `TotalDurationMetric`.
- Column naming follows the existing `patrol_` prefix on observations; leg-only measures use the `leg_` prefix.
- Tests are flat functions (`def test_x():`), no test classes. New tests go next to the existing tests for the module they cover.
- Commit at the end of every task.

## Review Focus

Inputs the spec implies but no happy-path test covers. Each has a pinned test in the owning task.

1. **A leg whose `end_time` is null (patrol still open).** Expected: the leg is skipped for observations (as today, via the `None in {subject_id, start}` check it already passes) and in the leg frame its end is clipped to `time_range.until`. Pinned in Task 2.
2. **A leg entirely outside the analysis window after clipping.** Expected: dropped from the leg frame, never a negative duration. Pinned in Task 2.
3. **`members` present but an empty list.** Expected: `leg_member_days` is 0, and an explode on `patrol_member_names` yields one row with `"None"`, not zero rows. Pinned in Tasks 2 and 5.
4. **More than 50 distinct subject ids across members and assets.** Expected: the subject lookup batches and every id resolves. Pinned in Task 4.
5. **A leg with no track segments under a spatial grouper.** Expected: it appears once in the `"None"` spatial group with its whole duration. Pinned in Task 6.

---

### Task 1: Carry leg fields onto patrol observations

**Files:**
- Modify: `ecoscope/io/earthranger.py:1189-1242` (`get_patrol_observations`)
- Test: `tests/test_earthranger_io.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: observations GeoDataFrame columns `patrol_segment_id: str` (always), and when present on the segment: `patrol_team: str | None`, `patrol_members: list[str]`, `patrol_assets: list[str]`, `patrol_segment_details: dict`, `patrol_type_details: dict`. Segments with `is_pause` true produce no observations.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_earthranger_io.py`, using `@patch("ecoscope.io.earthranger.EarthRangerIO.get_subject_observations")` and `@patch("ecoscope.io.earthranger.EarthRangerIO.get_patrol_types")`. Build `patrols_df` as a one-row DataFrame with `id`, `title`, `serial_number`, `state`, and `patrol_segments` holding two dicts: leg A with `id="seg-a"`, `leader={"id":"s1","name":"Asha"}`, `time_range` start/end, `patrol_type="foot"`, `team="t1"`, `members=["s1","s2"]`, `assets=["v1"]`, `segment_details={"weather":"dry"}`, `type_details={}`; leg B identical except `id="seg-b"`, `is_pause=True`. `get_subject_observations` returns a `Relocations` with two fixes. `get_patrol_types` returns a frame indexed by id with `value="foot"`.

```python
def test_get_patrol_observations_carries_leg_fields(obs_mock, pt_mock, er_io):
    relocs = er_io.get_patrol_observations(patrols_df, include_patrol_details=True)
    gdf = relocs.gdf
    assert set(gdf["patrol_segment_id"]) == {"seg-a"}
    assert gdf["patrol_team"].iloc[0] == "t1"
    assert gdf["patrol_members"].iloc[0] == ["s1", "s2"]
    assert gdf["patrol_assets"].iloc[0] == ["v1"]
    assert gdf["patrol_segment_details"].iloc[0] == {"weather": "dry"}
    assert gdf["patrol_type_details"].iloc[0] == {}

def test_get_patrol_observations_skips_pause_legs(obs_mock, pt_mock, er_io):
    er_io.get_patrol_observations(patrols_df, include_patrol_details=True)
    assert obs_mock.call_count == 1  # leg B (is_pause) never fetched

def test_get_patrol_observations_without_new_fields_adds_no_columns(obs_mock, pt_mock, er_io):
    # same patrols_df with team/members/assets/segment_details/type_details/is_pause keys removed
    gdf = er_io.get_patrol_observations(legacy_patrols_df, include_patrol_details=True).gdf
    assert "patrol_segment_id" in gdf.columns
    for col in ["patrol_team", "patrol_members", "patrol_assets", "patrol_segment_details", "patrol_type_details"]:
        assert col not in gdf.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_earthranger_io.py -k "leg_fields or pause_legs or no_columns" -v`
Expected: FAIL with `KeyError: 'patrol_segment_id'` and `assert 2 == 1`.

- [ ] **Step 3: Implement in `get_patrol_observations`**

Inside the segment loop, before the `None in {subject_id, patrol_start_time}` check, add `if patrol_segment.get("is_pause"): continue`. Inside the `include_patrol_details` block, after `patrol_subject`, set `observation.gdf["patrol_segment_id"] = patrol_segment.get("id")`, then for each `(api_key, col)` in `[("team","patrol_team"),("members","patrol_members"),("assets","patrol_assets"),("segment_details","patrol_segment_details"),("type_details","patrol_type_details")]`: if `api_key in patrol_segment`, assign the value to every row. Assign list and dict values with `pd.Series([value] * len(gdf), index=gdf.index)` so pandas does not broadcast a list across rows as separate scalars. Columns added only when present means the later `pd.concat` yields NaN for legs that lack a field a sibling leg has; that is acceptable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_earthranger_io.py -k "patrol_observations" -v`
Expected: PASS, including the pre-existing patrol observation tests.

- [ ] **Step 5: Commit**

```bash
git add ecoscope/io/earthranger.py tests/test_earthranger_io.py
git commit -m "feat(io): carry patrol leg fields onto observations and skip pause legs"
```

---

### Task 2: Leg frame

**Files:**
- Modify: `ecoscope/io/earthranger_utils.py` (after `unpack_events_from_patrols_df`, line 206)
- Modify: `ecoscope/platform/tasks/io/_earthranger.py` (after `unpack_events_from_patrols_df`, line 1049) and its combined-params variant block (lines 898-929)
- Modify: `ecoscope/platform/tasks/io/__init__.py` (import and `__all__`)
- Test: `tests/test_earthranger_utils.py`, `tests/platform/tasks/test_io_earthranger.py`

**Interfaces:**
- Consumes: `TimeRange` from `ecoscope.platform.tasks.filter._filter` (`since`, `until`, tz-aware).
- Produces:
  - `ecoscope.io.earthranger_utils.patrol_segments_from_patrols_df(patrols_df: pd.DataFrame, since: datetime | None = None, until: datetime | None = None) -> pd.DataFrame`
  - task `get_patrol_segments_from_patrols_df(patrols_df: PatrolsDF, time_range: TimeRangeAnnotation, truncate_to_time_range: TruncateToTimeRangeAnnotation = True, raise_on_empty: RaiseOnEmptyAnnotation = True) -> AnyDataFrame | EmptyDataFrame`
  - task `get_patrol_segments_from_patrols_df_and_combined_params(patrols_df: PatrolsDF, params: CombinedPatrolAndEventsParams) -> AnyDataFrame | EmptyDataFrame`
  - Leg frame columns, one row per non-pause leg: `patrol_id: str`, `patrol_serial_number: int`, `patrol_segment_id: str`, `patrol_type: str` (value slug, as `unpack_events_from_patrols_df` does), `patrol_subject: str`, `segment_start: datetime[UTC]`, `segment_end: datetime[UTC]`, `leg_active_seconds: float`, `leg_member_days: float`, `patrol_elapsed_seconds: float` (value on the earliest leg of each patrol, NaN on the rest), and when present on the leg: `patrol_team`, `patrol_members`, `patrol_assets`, `patrol_segment_details`, `patrol_type_details`.

- [ ] **Step 1: Write the failing tests in `tests/test_earthranger_utils.py`**

Reuse the `_patrols_df_with_events` helper pattern to build a patrols_df with one patrol (`id="p1"`, `serial_number=7`) and legs: leg 1 `08:00-12:00` members `["a","b","c"]`; leg 2 `12:00-13:00` `is_pause=True`; leg 3 `13:00-16:00` members `["a","b","c"]`; all on 2026-01-05 UTC, leader `{"id":"a","name":"Asha"}`, `patrol_type="foot"`.

```python
def test_patrol_segments_excludes_pauses_and_computes_durations():
    legs = patrol_segments_from_patrols_df(patrols_df)
    assert legs["patrol_segment_id"].tolist() == ["seg1", "seg3"]
    assert legs["leg_active_seconds"].tolist() == [4 * 3600.0, 3 * 3600.0]
    assert legs["leg_member_days"].tolist() == [3.0, 3.0]
    assert legs["patrol_elapsed_seconds"].tolist()[0] == 8 * 3600.0
    assert pd.isna(legs["patrol_elapsed_seconds"].iloc[1])
    assert legs["patrol_type"].iloc[0] == "foot"
    assert legs["patrol_subject"].iloc[0] == "Asha"
    assert legs["patrol_serial_number"].iloc[0] == 7

def test_patrol_segments_clips_to_window():
    legs = patrol_segments_from_patrols_df(patrols_df, since=T("2026-01-05T10:00Z"), until=T("2026-01-05T14:00Z"))
    assert legs["leg_active_seconds"].tolist() == [2 * 3600.0, 1 * 3600.0]
    assert legs["patrol_elapsed_seconds"].iloc[0] == 4 * 3600.0

def test_patrol_segments_drops_legs_outside_window():
    legs = patrol_segments_from_patrols_df(patrols_df, since=T("2026-01-05T14:30Z"), until=T("2026-01-05T15:00Z"))
    assert legs["patrol_segment_id"].tolist() == ["seg3"]
    assert (legs["leg_active_seconds"] >= 0).all()

def test_patrol_segments_open_leg_clips_end_to_until():
    # leg 3 has time_range end_time None
    legs = patrol_segments_from_patrols_df(open_patrols_df, until=T("2026-01-05T18:00Z"))
    assert legs["leg_active_seconds"].iloc[-1] == 5 * 3600.0

def test_patrol_segments_member_days_spans_midnight():
    # one leg 2026-01-05T22:00Z to 2026-01-06T02:00Z, members ["a","b"]
    assert patrol_segments_from_patrols_df(midnight_df)["leg_member_days"].iloc[0] == 4.0

def test_patrol_segments_without_members_gives_nan_member_days_and_no_new_columns():
    legs = patrol_segments_from_patrols_df(legacy_patrols_df)
    assert legs["leg_member_days"].isna().all()
    assert "patrol_members" not in legs.columns and "patrol_team" not in legs.columns

def test_patrol_segments_empty_members_gives_zero_member_days():
    assert patrol_segments_from_patrols_df(empty_members_df)["leg_member_days"].iloc[0] == 0.0

def test_patrol_segments_empty_patrols_df_returns_empty():
    assert patrol_segments_from_patrols_df(pd.DataFrame()).empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_earthranger_utils.py -k patrol_segments -v`
Expected: FAIL with `ImportError: cannot import name 'patrol_segments_from_patrols_df'`.

- [ ] **Step 3: Implement `patrol_segments_from_patrols_df` in `earthranger_utils.py`**

Walk `patrols_df` rows and `patrol_segments` as `unpack_events_from_patrols_df` does. Skip legs with `is_pause` true or with no `time_range.start_time`. Parse start/end with `pd.to_datetime(..., utc=True)`; a null end becomes `until` when given, else the leg is skipped. Clip start to `since` and end to `until` when given; drop legs where clipped end <= clipped start. `leg_active_seconds = (end - start).total_seconds()`. `leg_member_days = len(members) * number of distinct UTC dates in [start, end]` when `"members" in segment`, else NaN. After collecting rows, compute per `patrol_id`: `elapsed = max(end) - min(start)` in seconds, assign to the row with the earliest `segment_start`, NaN elsewhere. Optional columns are added only if the key appeared on at least one leg; legs lacking it get NaN. Return `pd.DataFrame` (no geometry); empty input returns `pd.DataFrame()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_earthranger_utils.py -k patrol_segments -v`
Expected: PASS.

- [ ] **Step 5: Write the failing task test in `tests/platform/tasks/test_io_earthranger.py`**

```python
def test_get_patrol_segments_from_patrols_df_task_validates_and_truncates():
    result = task(get_patrol_segments_from_patrols_df).validate().call(
        patrols_df=patrols_df,  # the three-leg fixture, as a DataFrame with id/state/serial_number/patrol_segments
        time_range={"since": "2026-01-05T10:00:00Z", "until": "2026-01-05T14:00:00Z", "timezone": UTC_TZ_DICT},
    )
    assert result["leg_active_seconds"].tolist() == [2 * 3600.0, 1 * 3600.0]

def test_get_patrol_segments_from_patrols_df_raise_on_empty():
    with pytest.raises(ValueError, match="No patrol segments"):
        get_patrol_segments_from_patrols_df(patrols_df=empty_patrols_df, time_range=tr, raise_on_empty=True)
```

- [ ] **Step 6: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_io_earthranger.py -k patrol_segments -v`
Expected: FAIL with ImportError.

- [ ] **Step 7: Implement the two tasks in `platform/tasks/io/_earthranger.py` and export them**

`get_patrol_segments_from_patrols_df` is `@register(tags=["io"])`, calls the utils function with `since=time_range.since, until=time_range.until` when `truncate_to_time_range` else no bounds, raises `ValueError("No patrol segments in provided patrols_df")` when empty and `raise_on_empty`. The combined-params variant follows the pattern at lines 898-929: `task(get_patrol_segments_from_patrols_df).validate().call(patrols_df=patrols_df, time_range=params.time_range, truncate_to_time_range=params.truncate_to_time_range, raise_on_empty=params.raise_on_empty)`. Add both to `platform/tasks/io/__init__.py` imports and `__all__`.

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_io_earthranger.py -k patrol_segments -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ecoscope/io/earthranger_utils.py ecoscope/platform/tasks/io/_earthranger.py ecoscope/platform/tasks/io/__init__.py tests/test_earthranger_utils.py tests/platform/tasks/test_io_earthranger.py
git commit -m "feat(io): add patrol leg frame task with clipped active and elapsed durations"
```

---

### Task 3: Leg metric presets and summary routing

**Files:**
- Modify: `ecoscope/platform/tasks/analysis/_patrol_summary.py:170-206`
- Modify: `ecoscope/platform/tasks/analysis/_summary.py` (after `summarize_df`, line 224)
- Modify: `ecoscope/platform/tasks/analysis/__init__.py`
- Test: `tests/platform/tasks/test_patrol_summary.py`

**Interfaces:**
- Consumes: leg frame columns from Task 2.
- Produces:
  - Preset classes `ActiveDurationMetric` (`metric="active_duration"`, `unit: Literal["h","d"]="h"`, title "Active Duration", column `leg_active_seconds`, sum, seconds to unit, 2 dp), `ElapsedDurationMetric` (`metric="elapsed_duration"`, same unit field, title "Elapsed Duration", column `patrol_elapsed_seconds`), `LegCountMetric` (`metric="leg_count"`, title "Leg Count", `nunique` of `patrol_segment_id`), `MemberDaysMetric` (`metric="member_days"`, title "Member-days", sum of `leg_member_days`, 0 dp). All appended to `PatrolSummaryMetric`.
  - `LEG_METRIC_COLUMNS: frozenset[str] = frozenset({"leg_active_seconds", "patrol_elapsed_seconds", "patrol_segment_id", "leg_member_days"})` in `_summary.py`, imported by `_patrol_summary.py`.
  - tasks in `_summary.py`: `select_track_metrics(summary_params: list[SummaryParam]) -> list[SummaryParam]`, `select_leg_metrics(summary_params: list[SummaryParam]) -> list[SummaryParam]`, `join_summary_tables(left: AnyDataFrame, right: AnyDataFrame) -> AnyDataFrame`.
  - `_DEFAULT_PATROL_SUMMARY_METRICS` is unchanged (the effort-table default list change belongs to the workflow spec, not the library).

- [ ] **Step 1: Write the failing tests**

```python
def test_set_patrol_summary_metrics_defaults_unchanged():
    assert [p.display_name for p in set_patrol_summary_metrics()] == [
        "Patrol Count", "Total Distance (km)", "Total Duration (hrs)", "Patrol Days",
        "Merged Area Covered (km²)", "Unmerged Area Covered (km²)",
    ]

def test_leg_presets():
    active, elapsed, legs, days = set_patrol_summary_metrics(
        [{"metric": "active_duration", "unit": "d"}, {"metric": "elapsed_duration"},
         {"metric": "leg_count"}, {"metric": "member_days"}]
    )
    assert (active.display_name, active.column, active.aggregator, active.new_unit) == ("Active Duration (days)", "leg_active_seconds", "sum", Unit.DAY)
    assert (elapsed.display_name, elapsed.column) == ("Elapsed Duration (hrs)", "patrol_elapsed_seconds")
    assert (legs.display_name, legs.column, legs.aggregator) == ("Leg Count", "patrol_segment_id", "nunique")
    assert (days.display_name, days.column, days.aggregator, days.decimal_places) == ("Member-days", "leg_member_days", "sum", 0)

def test_select_track_and_leg_metrics_partition():
    params = set_patrol_summary_metrics([{"metric": "patrol_count"}, {"metric": "active_duration"}, {"metric": "total_distance"}, {"metric": "leg_count"}])
    assert [p.display_name for p in select_track_metrics(params)] == ["Patrol Count", "Total Distance (km)"]
    assert [p.display_name for p in select_leg_metrics(params)] == ["Active Duration (hrs)", "Leg Count"]

def test_join_summary_tables_outer_on_group_key():
    left = pd.DataFrame({"Patrol Count": [2, 1]}, index=pd.Index(["A", "B"], name="patrol_type"))
    right = pd.DataFrame({"Leg Count": [3]}, index=pd.Index(["A"], name="patrol_type"))
    joined = join_summary_tables(left, right)
    assert list(joined.columns) == ["Patrol Count", "Leg Count"]
    assert joined.loc["B", "Leg Count"] != joined.loc["B", "Leg Count"]  # NaN

def test_join_summary_tables_with_empty_right_keeps_left():
    left = pd.DataFrame({"Patrol Count": [2]}, index=pd.Index(["A"], name="patrol_type"))
    assert join_summary_tables(left, pd.DataFrame()).equals(left)

def test_leg_metrics_end_to_end_on_leg_frame():
    legs = pd.DataFrame({
        "patrol_type": ["foot", "foot", "car"],
        "patrol_segment_id": ["s1", "s3", "s9"],
        "leg_active_seconds": [4 * 3600.0, 3 * 3600.0, 3600.0],
        "patrol_elapsed_seconds": [8 * 3600.0, float("nan"), 3600.0],
        "leg_member_days": [3.0, 3.0, 1.0],
    })
    params = set_patrol_summary_metrics([{"metric": "active_duration"}, {"metric": "elapsed_duration"}, {"metric": "leg_count"}, {"metric": "member_days"}])
    out = summarize_df(legs, select_leg_metrics(params), groupby_cols=["patrol_type"])
    assert out.loc["foot"].tolist() == [7.0, 8.0, 2, 6.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_patrol_summary.py -v`
Expected: the new tests FAIL with `ValidationError` on unknown `metric` tags and ImportError for the three new tasks; `test_set_patrol_summary_metrics_defaults_unchanged` passes already.

- [ ] **Step 3: Implement the presets and tasks**

In `_patrol_summary.py` add the four `BaseModel` classes following `TotalDurationMetric` exactly (same `unit` field, `labeled_units(Unit.HOUR, Unit.DAY)`, `original_unit=Unit.SECOND`); `MemberDaysMetric.to_summary_param()` returns `StatSummaryParam(display_name="Member-days", aggregator="sum", column="leg_member_days", decimal_places=0)`. Append them to the `Union` before `CustomMetric`. In `_summary.py` define `LEG_METRIC_COLUMNS` and add the three `@register()` tasks: the selectors filter by `getattr(p, "column", None) in LEG_METRIC_COLUMNS`. `join_summary_tables` returns `left` when `right.empty`, `right` when `left.empty`, else `left.join(right, how="outer")`. Export all three from `analysis/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_patrol_summary.py tests/platform/tasks/test_encounter_rate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ecoscope/platform/tasks/analysis tests/platform/tasks/test_patrol_summary.py
git commit -m "feat(analysis): add leg duration, leg count and member-days presets with track/leg routing"
```

---

### Task 4: Team and roster name resolution

**Files:**
- Modify: `ecoscope/io/earthranger.py` (after `get_patrol_types`, line 914)
- Modify: `ecoscope/platform/tasks/io/_earthranger.py` (after Task 2's tasks)
- Modify: `ecoscope/platform/tasks/io/__init__.py`
- Test: `tests/test_earthranger_io.py`, `tests/platform/tasks/test_io_earthranger.py`

**Interfaces:**
- Consumes: `EarthRangerIO._get`, `EarthRangerIO.get_subjects(id=..., include_inactive=True)`, `erclient.ERClientNotFound`.
- Produces:
  - `EarthRangerIO.get_patrol_teams() -> pd.DataFrame` with columns `id`, `name`; empty frame with those columns on 404.
  - task `add_patrol_team_names(df: AnyDataFrame, client: EarthRangerClient) -> AnyDataFrame` adding `patrol_team_name: str`.
  - task `add_patrol_roster_names(df: AnyDataFrame, client: EarthRangerClient) -> AnyDataFrame` adding `patrol_member_names` and `patrol_asset_names` (each a `list[str]` per row, or the string `"None"` when the source column is absent).

- [ ] **Step 1: Confirm the config response shape against mep-dev**

Run with the `EARTHRANGER_*` env vars set:

```bash
.venv/bin/python -c "
from ecoscope.io import EarthRangerIO; import os, json
io = EarthRangerIO(server=os.environ['EARTHRANGER_SERVER'], username=os.environ['EARTHRANGER_USERNAME'], password=os.environ['EARTHRANGER_PASSWORD'])
print(json.dumps(io._get('activity/patrols/config/default/'), indent=1)[:2000])"
```

Expected: a JSON object with a list of teams each carrying `id` and `name` (the schema doc says "every team including retired ones"). Record the exact key path (assumed `teams`) and use it in Step 4. If the route returns 404, the flag is off on mep-dev; stop and report.

- [ ] **Step 2: Write the failing client tests in `tests/test_earthranger_io.py`**

```python
@patch("ecoscope.io.earthranger.EarthRangerIO._get")
def test_get_patrol_teams(get_mock, er_io):
    get_mock.return_value = {"teams": [{"id": "t1", "name": "Alpha", "is_active": True}, {"id": "t2", "name": "Bravo", "is_active": False}]}
    teams = er_io.get_patrol_teams()
    assert teams[["id", "name"]].values.tolist() == [["t1", "Alpha"], ["t2", "Bravo"]]
    assert get_mock.call_args.args[0] == "activity/patrols/config/default/"

@patch("ecoscope.io.earthranger.EarthRangerIO._get", side_effect=ERClientNotFound())
def test_get_patrol_teams_404_is_empty(get_mock, er_io):
    teams = er_io.get_patrol_teams()
    assert teams.empty and list(teams.columns) == ["id", "name"]
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_earthranger_io.py -k patrol_teams -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 4: Implement `get_patrol_teams`**

Call `self._get("activity/patrols/config/default/")`, catch `ERClientNotFound` and return `pd.DataFrame(columns=["id", "name"])`; otherwise `pd.DataFrame(response[<teams key from Step 1>])`.

- [ ] **Step 5: Run to verify they pass, then write the failing task tests in `tests/platform/tasks/test_io_earthranger.py`**

Use a `MagicMock()` client. `get_patrol_teams.return_value` is the two-team frame; `get_subjects.return_value` is a frame with `id`/`name` for the ids requested (use `side_effect` that parses the comma-joined `id` kwarg so a 60-id request is answered).

```python
def test_add_patrol_team_names_resolves_and_falls_back():
    df = pd.DataFrame({"patrol_team": ["t1", "t9", None]})
    out = add_patrol_team_names(df, client)
    assert out["patrol_team_name"].tolist() == ["Alpha", "t9", "None"]

def test_add_patrol_team_names_without_column_fills_none():
    out = add_patrol_team_names(pd.DataFrame({"x": [1, 2]}), client)
    assert out["patrol_team_name"].tolist() == ["None", "None"]
    client.get_patrol_teams.assert_not_called()

def test_add_patrol_roster_names_resolves_lists():
    df = pd.DataFrame({"patrol_members": [["s1", "s2"], []], "patrol_assets": [["v1"], ["zz"]]})
    out = add_patrol_roster_names(df, client)
    assert out["patrol_member_names"].tolist() == [["Asha", "Ben"], []]
    assert out["patrol_asset_names"].tolist() == [["Land Cruiser"], ["zz"]]
    kwargs = client.get_subjects.call_args.kwargs
    assert kwargs["include_inactive"] is True
    assert set(kwargs["id"].split(",")) == {"s1", "s2", "v1", "zz"}

def test_add_patrol_roster_names_resolves_more_than_fifty_ids():
    ids = [f"s{i}" for i in range(60)]
    out = add_patrol_roster_names(pd.DataFrame({"patrol_members": [ids], "patrol_assets": [[]]}), client)
    assert out["patrol_member_names"].iloc[0] == [f"Name {i}" for i in range(60)]

def test_add_patrol_roster_names_without_columns_fills_none():
    out = add_patrol_roster_names(pd.DataFrame({"x": [1]}), client)
    assert out["patrol_member_names"].tolist() == ["None"] and out["patrol_asset_names"].tolist() == ["None"]
    client.get_subjects.assert_not_called()
```

- [ ] **Step 6: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_io_earthranger.py -k "team_names or roster_names" -v`
Expected: FAIL with ImportError.

- [ ] **Step 7: Implement the two tasks (`@register(tags=["io"])`) and export them**

`add_patrol_team_names`: if `"patrol_team"` not in columns, set `patrol_team_name = "None"` and return. Else fetch teams once, build `dict(zip(id, name))`, map with fallback to the id, and `"None"` for null. `add_patrol_roster_names`: for each of `patrol_members`, `patrol_assets` absent, set the name column to `"None"`; collect unique ids from the present list columns, call `client.get_subjects(id=",".join(ids), include_inactive=True)` once (the client batches internally at 50), map each list with fallback to the id. Rows whose list value is NaN get `[]`.

- [ ] **Step 8: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_io_earthranger.py -k "team_names or roster_names" tests/test_earthranger_io.py -k patrol_teams -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ecoscope/io/earthranger.py ecoscope/platform/tasks/io/_earthranger.py ecoscope/platform/tasks/io/__init__.py tests/test_earthranger_io.py tests/platform/tasks/test_io_earthranger.py
git commit -m "feat(io): resolve patrol team and roster display names"
```

---

### Task 5: Explode list columns for selected groupers

**Files:**
- Modify: `ecoscope/platform/tasks/transformation/_exploding.py`
- Modify: `ecoscope/platform/tasks/transformation/__init__.py`
- Test: `tests/platform/tasks/test_exploding.py`

**Interfaces:**
- Consumes: `AllGrouper | UserDefinedGroupers`, `ValueGrouper` from `ecoscope.platform.indexes`.
- Produces: task `explode_for_groupers(df: AnyDataFrame, groupers: AllGrouper | UserDefinedGroupers) -> AnyDataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
def test_explode_for_groupers_explodes_only_selected_list_columns():
    df = pd.DataFrame({"patrol_member_names": [["Asha", "Ben"], ["Chidi"]], "patrol_asset_names": [["LC"], ["LC"]], "dist_meters": [12000.0, 5000.0]})
    out = explode_for_groupers(df, [ValueGrouper(index_name="patrol_member_names")])
    assert out["patrol_member_names"].tolist() == ["Asha", "Ben", "Chidi"]
    assert out["dist_meters"].tolist() == [12000.0, 12000.0, 5000.0]
    assert out["patrol_asset_names"].tolist() == [["LC"], ["LC"], ["LC"]]

def test_explode_for_groupers_empty_list_keeps_one_row():
    df = pd.DataFrame({"patrol_member_names": [[]], "dist_meters": [1.0]})
    out = explode_for_groupers(df, [ValueGrouper(index_name="patrol_member_names")])
    assert len(out) == 1 and out["patrol_member_names"].tolist() == ["None"]

def test_explode_for_groupers_is_noop_for_all_grouper_scalar_and_absent_columns():
    df = pd.DataFrame({"patrol_type": ["a", "b"], "patrol_member_names": [["x"], ["y", "z"]]})
    assert explode_for_groupers(df, AllGrouper()).equals(df)
    assert explode_for_groupers(df, [ValueGrouper(index_name="patrol_type")]).equals(df)
    assert explode_for_groupers(df, [ValueGrouper(index_name="missing")]).equals(df)

def test_explode_for_groupers_preserves_multiindex_levels():
    df = pd.DataFrame({"patrol_member_names": [["a", "b"]], "v": [1]}).set_index(pd.Index(["2026-01"], name="TemporalGrouper_%Y-%m"), append=True)
    out = explode_for_groupers(df, [ValueGrouper(index_name="patrol_member_names")])
    assert out.index.names == df.index.names and len(out) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_exploding.py -k for_groupers -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `explode_for_groupers`**

Return `df` unchanged for `AllGrouper`. For each `ValueGrouper` whose `index_name` is a column whose non-null values are lists (`df[col].dropna().map(lambda v: isinstance(v, list)).any()`), call `df = df.explode(col)` (keeps the index, so MultiIndex levels survive), then `df[col] = df[col].where(df[col].notna(), "None")` so an empty list's single NaN row becomes `"None"`. Register and export.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_exploding.py tests/platform/tasks/test_groupby.py -v`
Expected: PASS.

- [ ] **Step 5: Add a split_groups integration test to `tests/platform/tasks/test_groupby.py`**

```python
def test_split_groups_after_explode_for_groupers_gives_full_credit():
    df = pd.DataFrame({"patrol_member_names": [["Asha", "Ben", "Chidi"]], "dist_meters": [12000.0]})
    groupers = [ValueGrouper(index_name="patrol_member_names")]
    groups = split_groups(explode_for_groupers(df, groupers), groupers)
    assert len(groups) == 3
    assert all(g["dist_meters"].sum() == 12000.0 for _, g in groups)
```

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_groupby.py -k full_credit -v`
Expected: PASS without further code changes (if `split_groups` raises on dtype, the `"None"` fill in Step 3 is wrong and must produce object dtype strings).

- [ ] **Step 6: Commit**

```bash
git add ecoscope/platform/tasks/transformation/_exploding.py ecoscope/platform/tasks/transformation/__init__.py tests/platform/tasks/test_exploding.py tests/platform/tasks/test_groupby.py
git commit -m "feat(transformation): explode list columns for selected categorical groupers"
```

---

### Task 6: Apportion leg metrics across spatial features

**Files:**
- Create: `ecoscope/platform/tasks/transformation/_leg_apportion.py`
- Modify: `ecoscope/platform/tasks/transformation/__init__.py`
- Test: `tests/platform/tasks/test_leg_apportion.py`

**Interfaces:**
- Consumes: leg frame (Task 2) and a trajectory frame that has been through `add_spatial_index` and carries a `patrol_segment_id` column (the workflow spec renames `extra__patrol_segment_id` to `patrol_segment_id` before this task; see Task 8).
- Produces: task `apportion_legs_by_spatial_index(legs: AnyDataFrame, indexed_traj: AnyGeoDataFrame, groupers: AllGrouper | UserDefinedGroupers) -> AnyDataFrame`. For each `SpatialGrouper` present in both `groupers` and `indexed_traj.index.names`, the result has that level appended to its index; `leg_active_seconds`, `leg_member_days`, `patrol_elapsed_seconds` are multiplied by the leg's share of `dist_meters` in that feature; `patrol_segment_id` is copied unchanged.

- [ ] **Step 1: Write the failing tests**

Fixtures: `legs` with two rows, `s1` (active 7200, member_days 2, elapsed 7200) and `s2` (active 3600, member_days 1, elapsed NaN). `indexed_traj` with `patrol_segment_id` `["s1","s1","s1"]`, `dist_meters` `[300.0, 100.0, 0.0]`, and index level `SpatialGrouper_regions` `["North","South","South"]` appended to a RangeIndex. `groupers = [SpatialGrouper(spatial_index_name="regions")]`.

```python
def test_apportion_splits_by_track_share():
    out = apportion_legs_by_spatial_index(legs, indexed_traj, groupers).reset_index()
    s1 = out[out["patrol_segment_id"] == "s1"].set_index("SpatialGrouper_regions")
    assert s1.loc["North", "leg_active_seconds"] == 5400.0 and s1.loc["South", "leg_active_seconds"] == 1800.0
    assert s1.loc["North", "leg_member_days"] == 1.5 and s1.loc["North", "patrol_elapsed_seconds"] == 5400.0
    assert out["leg_active_seconds"].sum() == 7200.0 + 3600.0

def test_apportion_leg_without_track_lands_in_none_group_whole():
    out = apportion_legs_by_spatial_index(legs, indexed_traj, groupers).reset_index()
    s2 = out[out["patrol_segment_id"] == "s2"]
    assert s2["SpatialGrouper_regions"].tolist() == ["None"] and s2["leg_active_seconds"].iloc[0] == 3600.0

def test_apportion_noop_without_spatial_grouper():
    assert apportion_legs_by_spatial_index(legs, indexed_traj, [ValueGrouper(index_name="patrol_type")]).equals(legs)
    assert apportion_legs_by_spatial_index(legs, indexed_traj, AllGrouper()).equals(legs)

def test_apportion_leg_count_counts_once_per_feature():
    out = apportion_legs_by_spatial_index(legs, indexed_traj, groupers)
    assert out.groupby(level="SpatialGrouper_regions")["patrol_segment_id"].nunique().to_dict() == {"North": 1, "South": 1, "None": 1}
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_leg_apportion.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement**

For each `SpatialGrouper sg` with `sg.index_name in indexed_traj.index.names`: `shares = indexed_traj.reset_index().groupby(["patrol_segment_id", sg.index_name])["dist_meters"].sum()`; divide by `shares.groupby(level=0).transform("sum")`; drop rows where the leg total is 0; `reset_index(name="share")`. Left-merge onto `legs` on `patrol_segment_id`; rows with no match get `share = 1.0` and `sg.index_name = "None"`. Multiply the three measure columns by `share` where present; drop `share`; `set_index(sg.index_name, append=True)`. Apply groupers in order so two spatial groupers stack. Register and export.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_leg_apportion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ecoscope/platform/tasks/transformation/_leg_apportion.py ecoscope/platform/tasks/transformation/__init__.py tests/platform/tasks/test_leg_apportion.py
git commit -m "feat(transformation): apportion leg metrics across spatial features by track share"
```

---

### Task 7: Trajectory carry-through of list and dict columns

**Files:**
- Test: `tests/platform/tasks/test_indexing.py` (or a new `tests/test_trajectory_extra_columns.py` if no trajectory test file fits)
- Modify only if the test fails: `ecoscope/trajectory.py:139-141`

**Interfaces:**
- Consumes: `Relocations`, `Trajectory.from_relocations` (`ecoscope/platform/tasks/preprocessing/_preprocessing.py:144`).
- Produces: a guarantee that `extra__patrol_segment_id`, `extra__patrol_members`, `extra__patrol_segment_details` reach the trajectory frame intact, taking the value of each segment's start relocation.

- [ ] **Step 1: Write the test**

Build a `Relocations` of three fixes for one `groupby_col` with columns `patrol_segment_id=["s1","s1","s1"]`, `patrol_members=[["a","b"]]*3`, `patrol_segment_details=[{"k":1}]*3`. Convert to a trajectory the way `relocations_to_trajectory` does.

```python
def test_trajectory_keeps_list_and_dict_extra_columns():
    traj = Trajectory.from_relocations(Relocations(gdf=relocs_gdf))  # as _preprocessing.py:144 does
    gdf = traj.gdf
    assert gdf["extra__patrol_segment_id"].tolist() == ["s1", "s1"]
    assert gdf["extra__patrol_members"].iloc[0] == ["a", "b"]
    assert gdf["extra__patrol_segment_details"].iloc[0] == {"k": 1}
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests -k keeps_list_and_dict_extra -v`
Expected: PASS. `Trajectory` joins extra columns by relocation index (`trajectory.py:139-141`), which already takes the start fix's values. If it FAILS, change the join at line 141 to take extra columns from the start relocation explicitly and rerun.

- [ ] **Step 3: Commit**

```bash
git add tests
git commit -m "test: pin list and dict extra columns through trajectory conversion"
```

---

### Task 8: Seed mep-dev with patrols that carry the new leg fields

Both the library's live tests (`-m io`) and the workflow live test cases need real patrols on mep-dev with team, members, assets, details and a pause leg. Nothing there has them today.

**Files:**
- Create: `scripts/seed_patrol_legs_mep_dev.py`
- Modify: `tests/platform/tasks/test_io_earthranger.py` (live test, marked `io`)

**Interfaces:**
- Consumes: `EarthRangerIO.get_patrol_teams` (Task 4), `post_subject`, `post_source`, `post_subjectsource`, `post_observations`, `post_patrol`, `post_patrol_segment` (`earthranger.py:1441-1687`; `post_patrol_segment` forwards extra kwargs into the payload).
- Produces: on mep-dev, a patrol type `ecoscope_leg_test`, two person subjects and one vehicle subject in groups the patrol configuration references, and at least two patrols dated 2026-01-05 and 2026-01-06 whose legs carry the six fields, one of them with a pause leg. The script is idempotent: it looks up existing objects by name before creating. It prints the created patrol serial numbers.

- [ ] **Step 1: Check the patrol configuration on mep-dev**

```bash
.venv/bin/python -c "
from ecoscope.io import EarthRangerIO; import os, json
io = EarthRangerIO(server=os.environ['EARTHRANGER_SERVER'], username=os.environ['EARTHRANGER_USERNAME'], password=os.environ['EARTHRANGER_PASSWORD'])
print(json.dumps(io._get('activity/patrols/config/default/'), indent=1)[:3000])
print(json.dumps(io._get('activity/patrols/config/default/resolved/'), indent=1)[:3000])"
```

Expected: at least one team and non-empty member and asset groups. If any is empty, stop: the configuration is set in the EarthRanger admin UI and has no write route. Report to the user what is missing (team names, member group, asset group) and resume once set. Record the team id, the member subject group id and the asset subject group id for Step 3.

- [ ] **Step 2: Write the script**

`scripts/seed_patrol_legs_mep_dev.py`, run as `.venv/bin/python scripts/seed_patrol_legs_mep_dev.py`, reading `EARTHRANGER_SERVER`, `EARTHRANGER_USERNAME`, `EARTHRANGER_PASSWORD`. Steps inside:

1. Ensure subjects `Ecoscope Leg Tester A`, `Ecoscope Leg Tester B` (subject_type `person`) and `Ecoscope Leg Vehicle` (subject_type `vehicle`) exist, via `get_subjects(name=...)` then `post_subject`; add A and B to the member group and the vehicle to the asset group found in Step 1 (via the `subjectgroups/{id}/` PATCH if `post_subject` does not accept `subject_groups`; check `post_subject` kwargs first).
2. Ensure each has a source and subjectsource (`post_source`, `post_subjectsource`).
3. Post observations for A every 10 minutes from 08:00 to 16:00 UTC on 2026-01-05, walking a 12 km straight line near mep-dev's existing data (reuse the bounding box of `tests/conftest.py` subject fixtures), and the same on 2026-01-06.
4. Ensure patrol type `ecoscope_leg_test` exists (`get_patrol_types(include_inactive=True)`; create through `_post("activity/patrols/types/", payload={"value": "ecoscope_leg_test", "display": "Ecoscope Leg Test"})` if absent).
5. Create patrol 1 with `post_patrol(priority=0, title="Ecoscope leg test 2026-01-05")`, then three legs through `post_patrol_segment(patrol_id=<patrol id>, patrol_segment_id=str(uuid4()), patrol_type="ecoscope_leg_test", tracked_subject_id=<A's id>, start_time="2026-01-05T08:00:00Z", end_time="2026-01-05T12:00:00Z", team=<team id>, members=[A, B], assets=[vehicle], segment_details={"weather": "dry"}, type_details={})` (`earthranger.py:1602`; the trailing kwargs are merged into the POST payload), a pause leg 12:00-13:00 with `is_pause=True`, and a leg 13:00-16:00. Create patrol 2 on 2026-01-06 with one leg 09:00-11:00 and no pause. If the flat `segments/` POST rejects a field with 400, the flag is off or the field name differs; print the response body and stop.
6. Set both patrols' `state` to `done` (PATCH `activity/patrols/{id}/` with `{"state": "done"}`) so the default `status=["done"]` filter returns them.
7. Print the two serial numbers.

- [ ] **Step 3: Run the script and verify through the read path**

Run: `.venv/bin/python scripts/seed_patrol_legs_mep_dev.py`
Then:

```bash
.venv/bin/python -c "
from ecoscope.io import EarthRangerIO; import os
io = EarthRangerIO(server=os.environ['EARTHRANGER_SERVER'], username=os.environ['EARTHRANGER_USERNAME'], password=os.environ['EARTHRANGER_PASSWORD'])
p = io.get_patrols(since='2026-01-05T00:00:00Z', until='2026-01-07T00:00:00Z', patrol_type_value=['ecoscope_leg_test'], status=['done'])
segs = [s for row in p.patrol_segments for s in row]
print(len(p), [ (s['id'][:8], s.get('is_pause'), s.get('team'), s.get('members'), s.get('assets')) for s in segs ])"
```

Expected: 2 patrols; 3 segments visible for patrol 1 with `is_pause` false or absent (the pause leg is hidden because `include_pauses` is not sent), team id, two member ids and one asset id populated. Rerunning the script creates nothing new.

- [ ] **Step 4: Add a live test**

In `tests/platform/tasks/test_io_earthranger.py`, marked `@pytest.mark.io`, using the `client` fixture:

```python
def test_live_patrol_legs_on_mep_dev(client):
    tr = TimeRange(since=T("2026-01-05T00:00:00Z"), until=T("2026-01-07T00:00:00Z"), timezone=UTC_TIMEZONEINFO)
    patrols = get_patrols(client=client, time_range=tr, patrol_types=["ecoscope_leg_test"], status=["done"])
    legs = get_patrol_segments_from_patrols_df(patrols_df=patrols, time_range=tr)
    assert len(legs) == 3
    assert legs["leg_active_seconds"].sum() == 9 * 3600.0
    assert legs["patrol_elapsed_seconds"].dropna().tolist() == [8 * 3600.0, 2 * 3600.0]
    named = add_patrol_roster_names(add_patrol_team_names(legs, client), client)
    assert named["patrol_member_names"].iloc[0] == ["Ecoscope Leg Tester A", "Ecoscope Leg Tester B"]
    assert named["patrol_team_name"].iloc[0] != named["patrol_team"].iloc[0]
```

Run: `.venv/bin/python -m pytest tests/platform/tasks/test_io_earthranger.py -k live_patrol_legs -m io -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/seed_patrol_legs_mep_dev.py tests/platform/tasks/test_io_earthranger.py
git commit -m "test: seed mep-dev with patrol legs carrying team, roster and pause fields"
```

---

### Task 9: Example-return fixtures and task registry check

**Files:**
- Create: `ecoscope/platform/tasks/io/get-patrol-segments-from-patrols-df.example-return.parquet`, `get-patrol-segments-from-patrols-df-and-combined-params.example-return.parquet`, `add-patrol-team-names.example-return.parquet`, `add-patrol-roster-names.example-return.parquet`

Only tasks tagged `io` are mocked in workflow test cases (`wt_task.testing.create_func_magicmock` docstring: "mocking I/O tasks"), so the transformation and analysis tasks need no fixture.

**Interfaces:**
- Consumes: the seeded mep-dev data from Task 8 and the four IO tasks above.
- Produces: one example-return parquet per new IO task, so workflow mock test cases can run.

- [ ] **Step 1: Generate the fixtures from mep-dev**

Write a one-off snippet (not committed) that runs the seeded time range through `get_patrols` → `get_patrol_segments_from_patrols_df` → `add_patrol_team_names` → `add_patrol_roster_names`, saving each task's output with `to_parquet` under `ecoscope/platform/tasks/io/<task-name-with-hyphens>.example-return.parquet` (the combined-params variant gets a copy of the leg frame). List columns must round-trip: confirm with `load_parquet` from `ecoscope.platform.mock_loaders` that `patrol_members` comes back as lists, not strings.

- [ ] **Step 2: Verify every new IO task loads its fixture through the mock helper**

```bash
.venv/bin/python -c "
from wt_task.testing import create_func_magicmock
for name in ['get_patrol_segments_from_patrols_df','get_patrol_segments_from_patrols_df_and_combined_params','add_patrol_team_names','add_patrol_roster_names']:
    m = create_func_magicmock('ecoscope.platform.tasks.io', name)
    print(name, type(m()).__name__, list(m().columns)[:6])"
```

Expected: `DataFrame` for each, with `patrol_segment_id` among the columns, no exception.

- [ ] **Step 3: Run the whole platform suite**

Run: `.venv/bin/python -m pytest tests/platform -m "not io" -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add ecoscope/platform/tasks/io
git commit -m "chore: add example-return fixtures for patrol leg IO tasks"
```

---

### Task 10: Documentation and release note

**Files:**
- Modify: `doc/source` (the platform tasks reference page that lists IO and analysis tasks; find it with `grep -rl set_patrol_summary_metrics doc/source`)
- Modify: `README.rst` only if it lists tasks.

- [ ] **Step 1: Document the new tasks and presets**

Add the seven tasks and four presets to the page found above, with one line each, and a short "Patrol legs" paragraph stating: pauses are never requested, the leg frame is summarized separately, the three name columns always exist, member and asset groupers use full credit, spatial groupers apportion by track share.

- [ ] **Step 2: Commit**

```bash
git add doc README.rst
git commit -m "docs: describe patrol leg metrics, groupers and seeding script"
```

- [ ] **Step 3: Release**

Releases are tag-driven (`dynamic = ["version"]`, tags like `v2.25.1`). Open a PR to upstream `master`; after merge the release tag is cut by the maintainers' process. Note the resulting version in the design doc's rollout section so patrol-effort-table (step 2 of the rollout) knows what to pin.
