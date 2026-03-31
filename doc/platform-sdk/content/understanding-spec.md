# Understanding spec.yaml

In [Getting Started](./getting-started.md) you scaffolded and ran a workflow. Now let's understand what you built. This page walks through the events-map-example `spec.yaml` line by line, introducing each [Workflow Toolkit (wt)](https://wt.readthedocs.io/) concept as it appears.

---

## The `id` field

```yaml
id: events_map_example
```

Every workflow has a unique identifier. The compiler uses this to name the generated package.

---

## `requirements`

```yaml
requirements:
  - name: "ecoscope-platform"
    version: ">=2.11.3,<3"
    channel: "https://repo.prefix.dev/ecoscope-workflows/"
```

The `requirements` block lists conda and/or pypi packages the compiled workflow needs at runtime. The `ecoscope-platform` package provides all the built-in tasks. When you write custom tasks in a separate package, you add it here too — see [Your First Custom Task](./tutorials/first-custom-task.md).

---

## `workflow` — The DAG

The `workflow` list is the heart of the spec. Each entry is a **task instance** — a call to a registered Python function with specific arguments. Let's walk through each one.

### `workflow_details` — Metadata

```yaml
- name: Workflow Details
  id: workflow_details
  task: set_workflow_details
```

This calls [`set_workflow_details`][ecoscope.platform.tasks.config.set_workflow_details].

Sets the workflow name and description shown in the configuration form in Ecoscope Desktop / Ecoscope Web. Because there is no `partial` block, the [`name` and `description` parameters][ecoscope.platform.tasks.config.set_workflow_details] become user-configurable form fields.

### `er_client_name` — Data source connection

```yaml
- name: Data Source
  id: er_client_name
  task: set_er_connection
```

This calls [`set_er_connection`][ecoscope.platform.tasks.io.set_er_connection].

Prompts the user to select an EarthRanger connection. Ecoscope Desktop / Ecoscope Web renders a data-source picker for this because the task's return type is a connection protocol type. See [Data Sources](./tutorials/data-sources.md).

### `time_range` — Introducing `partial`

```yaml
- name: Time Range
  id: time_range
  task: set_time_range
  partial:
    time_format: '%d %b %Y %H:%M:%S'
```

This calls [`set_time_range`][ecoscope.platform.tasks.filter.set_time_range].

This is your first encounter with **[`partial`](https://wt.readthedocs.io/en/latest/reference/spec-yaml/#partial)**. The `partial` block binds arguments at compile time — they are baked into the workflow and *not* shown in the Ecoscope Desktop / Ecoscope Web configuration form. Here, `time_format` is fixed, but the actual start/end times remain configurable because they are *not* listed under `partial`. You can see [the full signature of `set_time_range`][ecoscope.platform.tasks.filter.set_time_range] in the Reference to understand all the arguments available.

**Rule of thumb**: anything under `partial` is fixed; any argument in the function's signature not listed becomes a form field.

### `get_events_data` — Wiring tasks together

```yaml
- name: Event Types
  id: get_events_data
  task: get_events
  partial:
    client: ${{ workflow.er_client_name.return }}
    time_range: ${{ workflow.time_range.return }}
    event_columns:
      - id
      - time
      - event_type
      - event_category
      - serial_number
      - geometry
    raise_on_empty: false
    include_details: false
    include_updates: false
    include_related_events: false
    include_display_values: true
```

This calls [`get_events`][ecoscope.platform.tasks.io.get_events].

The **`${{ workflow.<id>.return }}`** expression is how you wire one task's output into another task's input. Here, `client` receives the return value of `er_client_name`, and `time_range` receives the return value of `time_range`. These expressions form the edges of the DAG.

The remaining parameters (`event_columns`, `raise_on_empty`, etc.) are all fixed via `partial`, so users cannot change them in the form.

### `events_colormap` — Transformation

```yaml
- name: Events Colormap
  id: events_colormap
  task: apply_color_map
  partial:
    df: ${{ workflow.get_events_data.return }}
    input_column_name: event_type
    colormap: tab20b
    output_column_name: event_type_colormap
```

This calls [`apply_color_map`][ecoscope.platform.tasks.transformation.apply_color_map].

Applies a matplotlib colormap to the `event_type` column. This is the parameter you changed in [Step 5](./getting-started.md#step-5-change-a-parameter-recompile-and-re-run) of Getting Started.

### `events_map_layer` — First encounter with `skipif`

```yaml
- name: Create map layer from grouped Events
  id: events_map_layer
  task: create_point_layer
  skipif:
    conditions:
      - any_is_empty_df
      - any_dependency_skipped
      - all_geometry_are_none
  partial:
    layer_style:
      fill_color_column: event_type_colormap
      get_radius: 5
    legend:
      label_column: event_type_display
      color_column: event_type_colormap
    tooltip_columns: null
    geodataframe: ${{ workflow.events_colormap.return }}
```

This calls [`create_point_layer`][ecoscope.platform.tasks.results.create_point_layer].

This task has an explicit **`skipif`** block. The `skipif` system lets you conditionally skip a task at runtime. Each condition is the name of a registered skip function (see [skip tasks reference](./reference/tasks/skip.md)). If *any* condition returns `True`, the task is skipped and produces a `SkipSentinel` instead of a normal return value.

This task has its own `skipif` block, which introduces us to the skip system. The `task-instance-defaults` section at the bottom of the spec sets default skip conditions for all tasks (we will explain this shortly). When a task provides its own `skipif`, it overrides those defaults entirely.

Here, `all_geometry_are_none` is added because point layers make no sense if the data has no geometry. The other two conditions (`any_is_empty_df`, `any_dependency_skipped`) are the same as the defaults — they are repeated here because this task specifies its own `skipif` block, which overrides the defaults entirely.

### `events_ecomap` — Visualization

```yaml
- name: Draw Ecomap from grouped Events
  id: events_ecomap
  task: draw_ecomap
  partial:
    title: null
    tile_layers:
      - layer_name: "TERRAIN"
    north_arrow_style:
      placement: top-left
    legend_style:
      title: Event Type
      format_title: false
      placement: bottom-right
    static: false
    max_zoom: 20
    widget_id: Events Map
    geo_layers: ${{ workflow.events_map_layer.return }}
```

This calls [`draw_ecomap`][ecoscope.platform.tasks.results.draw_ecomap].

Renders an interactive map as an HTML string. Notice that `widget_id` must match the widget title defined downstream — this links the map to the correct dashboard widget.

### `events_ecomap_html_url` — Introducing `${{ env.VAR }}`

```yaml
- name: Persist grouped Events Ecomap as Text
  id: events_ecomap_html_url
  task: persist_text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
    filename_suffix: v2
    text: ${{ workflow.events_ecomap.return }}
```

This calls [`persist_text`][ecoscope.platform.tasks.io.persist_text].

The **`${{ env.VAR }}`** syntax reads an environment variable at runtime. `ECOSCOPE_WORKFLOWS_RESULTS` points to the directory where output files are stored. Ecoscope Desktop and Ecoscope Web both set this variable automatically when running the workflow. While `${{ env.VAR }}` can theoretically resolve any environment variable, in practice, `ECOSCOPE_WORKFLOWS_RESULTS` is currently the only one intended for direct reference in `spec.yaml`.

### `events_map_widget` — `skipif: never`

```yaml
- name: Create grouped Events Map Widget
  id: events_map_widget
  task: create_map_widget_single_view
  skipif:
    conditions:
      - never
  partial:
    title: Events Map
    view: ["All", "=", "True"]
    data: ${{ workflow.events_ecomap_html_url.return }}
```

This calls [`create_map_widget_single_view`][ecoscope.platform.tasks.results.create_map_widget_single_view].

This task overrides the defaults with **`never`** — a special skip condition that always returns `False`. This means the widget creation task will *always* run, even if its upstream dependency was skipped. Why? Because widget tasks need to produce a result (possibly empty) so the dashboard can render correctly.

Don't worry if the skip system feels complex at first — in practice, you will rarely need to modify skip conditions. The defaults handle the common case, and widget tasks just need `skipif: never`.

The `view` parameter is a composite filter that identifies which grouper view this widget belongs to. In this ungrouped example, there is only one view: `["All", "=", "True"]`. This special value conveys that all of the data is displayed in a single group. For workflows that slice data into multiple views (by month, by species, etc.), see the [Groupers tutorial](./tutorials/groupers.md).

### `events_dashboard` — Dashboard assembly

```yaml
- name: Create Dashboard with Map Widgets
  id: events_dashboard
  task: gather_dashboard
  partial:
    details: ${{ workflow.workflow_details.return }}
    widgets:
      - ${{ workflow.events_map_widget.return }}
    groupers:
      - index_name: "All"
    time_range: ${{ workflow.time_range.return }}
```

This calls [`gather_dashboard`][ecoscope.platform.tasks.results.gather_dashboard].

`gather_dashboard` collects all widgets, grouper definitions, and the time range into a `Dashboard` object. The terminal node of the workflow is serialized to JSON by the execution engine and sent back to Ecoscope Desktop / Ecoscope Web for visualization. The `widgets` list order determines the `widget_id` used in `layout.json` (0-indexed).

---

## `task-instance-defaults`

```yaml
task-instance-defaults:
  skipif:
    conditions:
      - any_is_empty_df
      - any_dependency_skipped
```

You noticed `skipif` on several tasks above. Rather than repeating common conditions on every task, the `task-instance-defaults` block sets them once. Every task inherits these defaults unless it provides its own `skipif` block.

The two default conditions are:

- **`any_is_empty_df`** — Skip if any DataFrame input is empty.
- **`any_dependency_skipped`** — Skip if any upstream task was skipped (returned a `SkipSentinel`).

Individual tasks can override this entirely. For example, `events_map_widget` overrides with `never` to ensure it always runs.

Skip conditions are themselves registered functions — they follow the same `@register()` pattern as tasks. See the [wt skipif reference](https://wt.readthedocs.io/en/latest/reference/spec-yaml/#skipif) for the full syntax.

Why defaults? Many tasks are not designed to handle empty inputs gracefully. Rather than raising errors when no data is available (for example, when the user's time range contains no events), the default skip conditions allow the workflow to gracefully skip those tasks and produce a dashboard with empty widgets instead of crashing.

---

## Key takeaways

| Concept | What it does |
|---------|-------------|
| `partial` | Binds arguments at compile time — they become fixed, not form fields |
| `${{ workflow.<id>.return }}` | Wires one task's output to another task's input |
| `${{ env.VAR }}` | Reads an environment variable at runtime |
| `skipif` | Conditionally skips a task based on runtime conditions |
| `never` | A skip condition that always returns False — overrides defaults |
| `task-instance-defaults` | Sets properties inherited by all tasks unless overridden |

For the full `spec.yaml` syntax, see the [wt spec.yaml Reference](https://wt.readthedocs.io/en/latest/reference/spec-yaml/).

---

## Try it yourself

Now that you understand the spec, try a small modification before moving on:

1. Change the `tile_layers` in `draw_ecomap` from `TERRAIN` to `SATELLITE` or `OPENSTREETMAP`.
2. Or change the `colormap` in `apply_color_map` to a different [matplotlib palette](https://matplotlib.org/stable/gallery/color/colormap_reference.html).

Recompile and see your change in Ecoscope Desktop. Each small edit reinforces the development loop.

---

## Next steps

- **[Tutorials](./tutorials.md)** — Learn to write custom tasks, build widgets, configure groupers, and more.
- **[Built-in Tasks](./built-in-tasks.md)** — Find the right built-in task for your use case.
