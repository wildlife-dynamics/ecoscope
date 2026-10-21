# Patrol Metrics Schema Update Design

2026-10-20. Canonical copy: https://claude.ai/artifact/CaEvhS5ZPwJbHApTCXde5F. Builds on the Patrol Metrics Survey and Patrol Schema Change docs from the same workspace.

## Intent

EarthRanger patrol legs now carry team, members, assets, two free-form details objects and a pause flag, behind the `patrol_schemas` preview flag (on at mep-dev). Ecoscope's patrol ingest reads only leader, time range and patrol type per leg, so none of this reaches any dashboard.

After this work a patrol dashboard user sees three things:

1. **More groupers.** Effort tables, encounter rates and charts can be broken down by team, member or asset, with display names rather than ids.
2. **Leg-specific metrics.** Active Duration, Elapsed Duration, Leg Count and Member-days join the existing preset list.
3. **Corrected duration.** Today's Total Duration is a sum of tracker-derived segment durations. It only approximates active patrol time, and only because the API hides pause legs. The new durations come from leg time ranges and treat pauses explicitly.

Sites that have not enabled `patrol_schemas` keep seeing exactly what they see today.

**Terminology.** A **leg** is an EarthRanger patrol segment: one entry in `patrol_segments` with a leader, a time range, a patrol type and the six new fields. It has no coordinates. A **track segment** is a trajectory segment ecoscope builds between two consecutive tracker fixes, carrying `dist_meters`, `timespan_seconds` and a geometry. One leg yields many track segments. Where this document says segment without qualification it means track segment.

## Goals, non-goals, anti-goals

**Goals**

- Team, member and asset available as categorical groupers wherever patrol workflows already offer patrol type.
- Active Duration, Elapsed Duration, Leg Count and Member-days available as metric presets.
- Per-member and per-asset rows carry full credit: a 12 km leg walked by three members shows 12 km on each member's row.
- Pause legs never contribute observations, distance or active time.
- One shared spec runs unchanged against a site with the flag on and a site with it off.

**Non-goals**

- Pause Count and Pause Duration metrics. Pauses are handled in ingest only.
- Metrics built on `segment_details` or `type_details`. The free-form objects are ingested and kept as columns but no preset reads them.
- mt-patrols and patrol-grid-maps. mt-patrols has its own SQL roster join and DOCX output; patrol-grid-maps is a draft.
- Fixing the other survey inconsistencies (unweighted average speed, grid CRS distance inflation, duplicated `set_traj_filters`) unless a change here touches the same line.

**Anti-goals**

- The value of an existing metric changing at a site that has not enabled the flag. Adding a new column to a default view is allowed; changing an existing number is not.
- A metric or grouper that errors when its column is absent. Absent must mean empty, never a failed run.
- Observations fetched for pause legs, which would inflate distance and duration.

## Constraints

- **Flag detection is per site.** `GET /api/v1.0/status/` returns `preview_features.patrol_schemas`. When it is off, the six leg fields are absent from responses, `config/default/` returns 404, and `include_pauses` on a nested patrol write is ignored. Ingest must treat missing fields as the normal case, not an error.
- **Branches.** All library work branches from upstream `master` of wt-tasks ecoscope (v2.25.1), not from `dashbuilder-poc`. Workflow work branches from `main`, and `ecoscope-web` must match `main` first. As of 2026-10-20 it does not: patrol-effort-table, patrol-encounter-rate-stats and patrol-encounter-rate-map each have one onboarding recompile commit on `ecoscope-web` that is not on `main`, wt-download-patrols has two, and `patrols` is 46 ahead and 5 behind. Syncing those is a precondition for each workflow's step in the rollout. patrol-effort-table also has an open draft PR #8 (grouper-style aggregator) that touches the same grouper code; it lands or closes before step 2.
- **Shared specs.** The same compiled workflow is deployed to many sites. Backward compatibility means column presence drives behaviour, not a per-site configuration switch.
- **Ids, not names.** `team` returns a bare uuid; `members` and `assets` return uuid arrays. Retired teams appear only in `config/default/`, and subjects no longer in a group do not appear in the resolved roster. The lookup must tolerate unresolved ids and fall back to the id string.
- **Site-wide segment schema needs a permission no set grants yet** (`activity.view_segmenttype`). Nothing here depends on that route.
- **Advisory validation.** The details objects can hold anything. They are kept as JSON-string columns and not interpreted.
- **Pin spread.** Workflows pin ecoscope-platform from 2.15 to 2.23. Each workflow that adopts the new presets must bump to the library release that carries them.
- **Existing preset names are stable.** Config forms and test cases reference preset names. No preset is renamed or removed; new ones are added.

## Ingest changes

All in `wt-tasks/ecoscope`. Line references are to v2.25.1.

**1. Carry the leg fields onto observations.** `get_patrol_observations` in `ecoscope/io/earthranger.py` (1168-1260) reads `leader`, `time_range` and `patrol_type` per segment and nothing else. It should also read the segment `id` and, when present, `is_pause`, `team`, `members`, `assets`, `segment_details` and `type_details`, using `.get()` so a flag-off site produces no column. New columns on the observations frame, following the existing `patrol_` prefix: `patrol_segment_id`, `patrol_team`, `patrol_members` (list), `patrol_assets` (list), `patrol_segment_details` (JSON string), `patrol_type_details` (JSON string). The leg's time range is already present as `patrol_start_time` and `patrol_end_time`. Any segment with `is_pause` true is skipped before observations are fetched, the same way segments with a missing leader or start are skipped today (1205). `PatrolObservationsGDF` and `RelocationsGDFSchema` are non-strict Pandera models, so extra columns pass validation unchanged.

**2. Do not request pauses.** `get_patrols` (916-1004) sends no `include_pauses`, so the API omits pause legs from nested `patrol_segments`, as the web app confirms. This design keeps it that way. Pause legs are not needed for any agreed metric: Active Duration sums the visible legs, Elapsed Duration spans first start to last end regardless of what sits between. The `is_pause` skip in step 1 is a guard for a future caller that does pass the flag. The warehouse path (`earthranger_utils.py:241-270`) rebuilds segments from flat columns and would need the same fields added when the warehouse exposes them; until then it yields no new columns, which is the flag-off behaviour.

**3. A leg frame.** A new task `get_patrol_segments_from_patrols_df(patrols_df, time_range)` returns one row per non-pause leg: `patrol_id`, `patrol_serial_number`, `patrol_segment_id`, `patrol_type`, `patrol_subject`, `patrol_team`, `patrol_members`, `patrol_assets`, `segment_start`, `segment_end` clipped to the time range, `leg_active_seconds`, `leg_member_days`, and `patrol_elapsed_seconds` populated on one row per patrol (its first leg) and NaN elsewhere. It mirrors `unpack_events_from_patrols_df` (`earthranger_utils.py:169-206`), which already walks the nested segments. The duration presets read these columns, so a leg with no surviving trajectory segments still counts.

**4. Summarize legs separately, then join.** Leg rows must not be concatenated into the trajectory frame: leg rows carry `patrol_id` and a start date whether or not the flag is on, so Patrol Count and Patrol Days would change at a flag-off site whenever a patrol has legs but no tracked segments, which breaks the first anti-goal. Instead `summarize_df` runs twice with the same `groupby_cols`, once on the trajectory frame for the existing presets and once on the leg frame for the four leg presets, and the two results are joined on the group key with an outer join. The leg frame carries the same grouper columns (`patrol_type`, `patrol_subject`, `patrol_serial_number`, the three name columns) so categorical groupers line up. For a temporal grouper the leg frame gets its own `add_temporal_index` on `segment_start`. Existing preset values are untouched by construction. The `set_patrol_summary_metrics` task keeps one metric list; a small splitter routes each selected preset to the frame it reads.

**5. Name resolution.** Two lookups, both optional:

- Teams: new client method calling `activity/patrols/config/default/`, which includes retired teams. On 404 it returns an empty frame. A task `add_patrol_team_names(df, client)` adds `patrol_team_name`, falling back to the id.
- Members and assets: the existing `get_subjects(id=...)` (`earthranger.py:141-243`) already batches comma-joined ids 50 at a time and returns `name`. A task collects the unique ids across `patrol_members` and `patrol_assets`, resolves them once with `include_inactive=True` so a subject retired since the patrol still gets its name (the schema doc confirms members and assets are subject ids, not user ids), and adds `patrol_member_names` and `patrol_asset_names` as lists. Unresolved ids fall back to the id string.

**6. Explode before grouping.** `split_groups` (`platform/tasks/groupby/_groupby.py:114-126`) raises on a non-string column, so a list column cannot be a grouper directly. A new task `explode_for_groupers(df, groupers)` explodes any list-valued column named by a selected `ValueGrouper` and leaves the frame untouched otherwise. Specs insert it immediately before `split_groups` and before `summarize_df`'s grouping. Because groupers are chosen at run time, the explode is conditional at run time, which is what keeps ungrouped totals free of duplication. The exploded frame is then grouped by the same column name, so `patrol_member_names` becomes a scalar string per row and passes the dtype check.

**7. Trajectory carry-through.** `Trajectory` prefixes non-core relocation columns with `extra__` (`ecoscope/trajectory.py:139-141`) and effort-table renames them back (`spec.yaml:378-394`). List and dict columns must survive that conversion; the builder verifies this and, if a segment built from two relocations drops or mangles them, takes the value from the segment's start relocation.

## Metrics

Four presets are added to `_patrol_summary.py`. All existing presets keep their name and formula. The three durations are kept distinct so a user can pick the one their report needs.

| Preset | Formula | Needs new schema |
| --- | --- | --- |
| Active Duration | sum over non-pause legs of leg end minus leg start, clipped to the analysis time range; s to h or d, 2 decimals | No, but pause-aware only when `is_pause` is present |
| Elapsed Duration | per patrol, last leg end minus first leg start, clipped to the analysis time range; summed over patrols in the group | No |
| Leg Count | unique leg id, pause legs excluded | No |
| Member-days | per non-pause leg, number of members on the row times calendar days the leg touches; summed. After a member explode the row holds one member | Yes, NaN when `members` absent |
| Total Duration (existing) | sum of trajectory `timespan_seconds`, unchanged | No |

Worked example. A patrol runs 08:00 to 16:00 with a pause 12:00 to 13:00 and three members. Total Duration gives whatever the leader's tracker covered, roughly 7 h. Active Duration gives 7 h. Elapsed Duration gives 8 h. Leg Count gives 2. Member-days gives 3.

**Why leg time ranges, not track segments.** Track-based duration shrinks when a tracker reports patchily and depends on the API hiding pauses. Leg time ranges are what the ranger recorded and are the same input the EarthRanger web app uses for its summary stats, so the two agree.

**Clipping.** Legs that overlap the analysis window are fetched (`patrols_overlap_daterange=True`), so a leg's time range is clipped to the window before summing. This also narrows the mismatched-windows problem the survey flagged, for the two new durations. Events are already truncated the same way.

**Where the leg columns come from.** The four new presets read leg rows from the leg frame (Ingest step 3), summarized separately from trajectory rows (step 4). `leg_active_seconds`, `leg_member_days` and `patrol_elapsed_seconds` live only on the leg frame, so a `sum` sees only legs; Leg Count is `nunique` of `patrol_segment_id`, which is robust to a leg with no surviving trajectory segments. No dedupe inside the preset is needed.

**Default sets.** patrol-effort-table's default metric list gains Active Duration. The other defaults are unchanged.

## Groupers

Three new categorical groupers: `patrol_team_name`, `patrol_member_names`, `patrol_asset_names`. They work everywhere `patrol_type` already does as a grouper: dashboard `split_groups`, the "Summarize by" grouper in effort-table and encounter-rate-stats, and chart category breakdowns.

**Full credit on explode.** `members` and `assets` are arrays per leg. Before grouping by member or asset, rows are exploded so one leg appears once per member. Every metric on a per-member row is computed on that member's copy of the leg, so each member is credited the leg's full distance, duration, events and patrol count. Column totals across members therefore exceed patrol totals, by design. Patrol totals belong on the ungrouped or team-grouped view, where there is no explode. Two presets need a stated rule: Member-days on an exploded row counts that one member, so a three-member leg shows 1 member-day per member row and 3 on the ungrouped row. Elapsed Duration sits on a patrol's first leg, so under a member grouper it is credited in full to each member of that first leg and not to a member who joined only on a later leg.

Explode happens only when a member or asset grouper is selected. No other grouper, and no ungrouped summary, ever sees exploded rows.

**Spatial groupers and leg metrics.** A leg has no geometry, so under a spatial grouper it inherits locations from its track segments. After `add_spatial_index` has placed each track segment in a feature, the share of each leg's total `dist_meters` falling in each feature is computed, and the leg row is replicated once per feature it touches with `leg_active_seconds`, `leg_member_days` and `patrol_elapsed_seconds` multiplied by that share. Durations therefore add up to the patrol total across features. Leg Count cannot be split: a leg counts once in every feature it touches. A leg with no track segments has no share and lands in the None group. Under a non-spatial grouper no split happens and the leg row is used whole. This replication happens inside the leg summarize path only (Ingest step 4) and never touches the trajectory frame.

**Name resolution.** `team` is a uuid; `members` and `assets` are uuid arrays. Teams resolve once per run through `config/default/`, which lists retired teams too. Members and assets resolve through the existing subjects lookup by id. Both are described in Ingest step 5. Ids that do not resolve fall back to the id string so the row still appears. When the config route returns 404 (flag off) there are no team, member or asset columns to resolve and the name columns are absent.

**Absent columns.** The name-resolution tasks always emit `patrol_team_name`, `patrol_member_names` and `patrol_asset_names`. When the source id columns are absent (flag off) they are filled with the string `None`, matching how `split_groups` already fills NA. So selecting a member grouper at a flag-off site yields one group named None rather than the `neither a column nor an index` error. The form keeps offering the groupers; hiding them per site is not possible with a shared spec.

**Patrol type.** Legs write `patrol_type` as the type's value. Workflows disagree on whether the column holds display name or value. This design does not change that, but every workflow touched in the rollout should expose the display name for grouping and keep the value available, following effort-table's current behaviour.

## Rollout order

1. **Library, wt-tasks ecoscope.** Ingest reads the new leg fields, roster fetch task, explode support in the categorical grouper path, four new presets. Released as a new ecoscope-platform version. Library tests cover both flag states.
2. **patrol-effort-table.** First consumer. Bumps the pin, adds the roster task to its spec, exposes the three groupers and four presets in the config form, adds Active Duration to the default metric list. Its mock test case carries the flag-on fixture; one live test case runs against a flag-off site.
3. **The remaining patrol workflows**, in this order, each bumping its pin and adopting what applies:
    - patrol-encounter-rate-stats: groupers and presets.
    - patrol-effort-plot and patrol-trend-plot: groupers and presets as chart metrics and breakdowns. Identical DAGs, so one change applied twice.
    - patrols: groupers for the single-value views and charts. Its hand-wired sum/mean chains are left as they are.
    - patrol-encounter-rate-map and patrol-track-density-map: team grouper for the per-group map views. Grid maths unchanged.
    - patrol-trajectory-event-map and wt-download-patrols: new leg columns pass through to the map tooltips and the download. No metrics.

Excluded: mt-patrols and patrol-grid-maps.

## Workflow wiring contract

What patrol-effort-table's spec adds once the library release is pinned, in DAG order. The other patrol workflows adopt the subset that applies. Task and column names are the ones the library plan's Interfaces blocks define; a mismatch fails at compile or in the mock test case.

| Order | Task (library) | Input | Output columns it adds or needs |
| --- | --- | --- | --- |
| 1 | `get_patrols_from_combined_params` (existing) | params | `patrols_df` |
| 2 | `get_patrol_observations_from_patrols_df_and_combined_params` (existing, now carries leg fields) | patrols_df | observations with `patrol_segment_id`, and when present `patrol_team`, `patrol_members`, `patrol_assets`, `patrol_segment_details`, `patrol_type_details` |
| 3 | `get_patrol_segments_from_patrols_df_and_combined_params` (new) | patrols_df, params | leg frame: `patrol_segment_id`, `segment_start`, `segment_end`, `leg_active_seconds`, `leg_member_days`, `patrol_elapsed_seconds`, plus the grouper columns |
| 4 | `relocations_to_trajectory` (existing) | observations | track frame with `extra__patrol_*` columns |
| 5 | `map_columns` (existing) | track frame | rename `extra__patrol_segment_id`, `extra__patrol_team`, `extra__patrol_members`, `extra__patrol_assets` to the unprefixed names, alongside today's renames |
| 6 | `add_patrol_team_names`, `add_patrol_roster_names` (new) on track frame and leg frame, each; `add_patrol_type_display_names` (new) on the leg frame | track frame and leg frame | `patrol_team_name`, `patrol_member_names`, `patrol_asset_names` on both frames; leg frame `patrol_type` becomes the display name, value kept in `patrol_type__value` |
| 7 | `convert_column_values_to_string` (existing) | track frame and leg frame, each | the shared group-key columns `patrol_serial_number`, `patrol_type`, `patrol_subject`, `patrol_team_name` are string-cast on BOTH frames (the leg frame's `patrol_type` is already the display name from step 6, before any grouping); never the two list columns |
| 8 | `add_temporal_index`, `add_spatial_index` (existing) | track frame | as today; plus `add_temporal_index` on the leg frame with `time_col: segment_start` |
| 9 | `apportion_legs_by_spatial_index` (new) | leg frame, indexed track frame, groupers | spatial index level on the leg frame |
| 10 | `explode_for_groupers` (new) | each frame, immediately before `split_groups` and before `summarize_df` | exploded rows when a member or asset grouper is selected |
| 11 | `set_patrol_summary_metrics` (existing) then `select_track_metrics`, `select_leg_metrics` (new) | form metric list | two parameter lists |
| 12 | `summarize_df` twice (existing), then `join_summary_tables` (new) | track frame, leg frame | one summary table; both `summarize_df` calls use the same `reset_index` setting (index join when False, merge on the shared group-key columns when True) |

**Column types.** `patrol_segment_details` and `patrol_type_details` are JSON strings on both frames (dict columns cannot be written to parquet). Member-days is blank (NaN), not 0, where a site has no member data.

**Form changes.** `$defs.ValueGrouper.oneOf` gains `patrol_team_name` (Team), `patrol_member_names` (Member), `patrol_asset_names` (Asset). `uiSchema.items.oneOf` for the metric list gains the four presets. The default metric list gains `{metric: active_duration, unit: h}`.

**Shared verification.** The library's live test and the workflow's live test case both read the seeded mep-dev patrols from 2026-01-05 and 2026-01-06 and assert the same numbers: 3 legs, 9 h active, 8 h and 2 h elapsed. The workflow mock test case loads the example-return parquets the library ships for the four new IO tasks.

## Left to the builder

- How the metric list is split between the trajectory and leg summarize calls (Ingest step 4): a routing task that takes the user's list and returns two, or a flag on each preset class. Either keeps one form field.
- Exact column names for the leg frame beyond the prefixes given here.
- How the per-patrol `patrol_elapsed_seconds` row is assigned a group when a temporal grouper splits a patrol's legs across periods. Default: the first leg's period.
- Where `explode_for_groupers` sits relative to `add_temporal_index` and `add_spatial_index` so the index levels survive the explode, and where the spatial share of each leg is computed from the indexed track segments before the leg summarize.
- Mock-io fixtures: the library tests cover both a patrols response with the six fields and one without. Workflow mock test cases use one example-return parquet per task, so they carry the flag-on fixture only; flag-off behaviour at the workflow level is proven by one live test case against a site without the flag.
- The form `$defs.ValueGrouper` `oneOf` list in each spec (`patrol-effort-table/spec.yaml:124-145`) gains the three name columns, and `uiSchema.items.oneOf` gains the four presets (spec.yaml:152-159). The `traj_rename_cols` and `traj_cols_to_string` steps gain the new columns, except that the two list columns are not stringified: `explode_for_groupers` runs after that cast and needs real lists.
- The user guide for each touched workflow describes the new groupers and the three durations.

**Calls made in this design, not agreed in conversation**

- Pauses are not requested from the API; `is_pause` is only a defensive skip.
- Teams resolve through `config/default/` rather than `config/default/resolved/`, so retired teams keep their names.
- Members and assets resolve through the existing subjects lookup rather than the config route.
- Clipping leg time ranges to the analysis window.
- Active Duration joins the effort-table default metric list.
- Grouper column names `patrol_team_name`, `patrol_member_names`, `patrol_asset_names`.
- Leg presets are summarized on a separate leg frame and joined to the trajectory summary, so existing preset values cannot move at a flag-off site.
- The three name columns always exist, filled with None when the flag is off.
- Member-days counts one member per exploded row; Elapsed Duration is credited to the members of a patrol's first leg.
- Subject lookup uses include_inactive so retired subjects still resolve.
- Flag-off workflow behaviour is proven by a live test case, not a mock fixture.
