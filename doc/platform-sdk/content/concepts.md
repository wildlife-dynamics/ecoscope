# Concepts

This page introduces the core ideas behind the Ecoscope Platform SDK. Read it once to build a mental model, then move on to [Getting Started](./getting-started.md) for hands-on work.

---

## The Ecoscope Ecosystem

Three layers work together:

**ecoscope (core library)** — Geospatial analysis primitives: relocations, trajectories, time-density, EcoMap, EcoPlot. If you need a low-level algorithm, it probably lives here.

**Ecoscope Platform SDK (`ecoscope.platform`)** — A curated set of typed *tasks*, *schemas*, *connections*, *groupers*, and *widgets* built on top of the core library. The Platform SDK is what you import when writing workflow specs. It ships ~80 ready-to-use tasks (see [Built-in Tasks](./built-in-tasks.md)).

**Workflow Toolkit (wt)** — A generic, framework-level toolkit for declaring, compiling, and running workflows. wt knows nothing about ecology — it provides `spec.yaml` syntax, the `@register()` decorator, the compiler, and runtime operators like `map`, `mapvalues`, and `skipif`. The Platform SDK plugs into wt. For deep dives into the framework, see the [wt documentation](https://wt.readthedocs.io/en/latest/concepts/).

**Ecoscope Desktop / Ecoscope Web** — End-user applications that load compiled workflow templates, present configuration forms, execute workflows, and display dashboards. You write the spec; Desktop runs it.

---

## The Development Loop

The core cycle looks like this:

```
edit spec.yaml  →  compile  →  load into Desktop  →  run  →  observe  →  iterate
```

**`spec.yaml`** is the single source of truth for your workflow. It declares which tasks to run, how they connect, and which parameters are exposed to the user.

**`wt-compiler compile`** reads `spec.yaml` and produces a standalone, installable package — complete with a `pixi.toml`, generated Python entry point, and a DAG visualization (`graph.png`).

Two companion files sit alongside `spec.yaml`:

- **`layout.json`** — Controls the dashboard grid layout (widget sizes and positions).
- **`test-cases.yaml`** — Supplies mock inputs so you can run the workflow locally without a live data source.

The [Getting Started](./getting-started.md) guide walks you through this cycle end-to-end.

---

## Anatomy of a Workflow

A workflow is a **directed acyclic graph (DAG)** of tasks, declared in the `workflow` list inside `spec.yaml`. Each entry is a *task instance*:

```yaml
- name: Events Colormap        # Display name in Desktop
  id: events_colormap           # Unique identifier for references
  task: apply_color_map         # The registered task function to call
  partial:                      # Arguments bound at compile time
    df: ${{ workflow.get_events_data.return }}
    colormap: tab20b
```

Key ideas:

- **`id`** — A unique identifier. Other tasks reference this task's output via `${{ workflow.<id>.return }}`.
- **`task`** — The name of a Python function decorated with `@register()`. This is how `spec.yaml` maps to code.
- **`partial`** — Values you fix at compile time. Anything listed under `partial` is *not* exposed as a form field in Desktop. Anything *not* listed becomes a configurable form field. This is how you control what users can change.
- **`${{ workflow.<id>.return }}`** — An expression that wires one task's output into another task's input, forming the edges of the DAG.

The **`task-instance-defaults`** block lets you set properties (like `skipif` conditions) once for all tasks, rather than repeating them on each instance. Individual tasks can override these defaults.

Some tasks run once per *group* via the `map` or `mapvalues` operators — these are explained in the [Groupers tutorial](./tutorials/groupers.md).

For a concrete, line-by-line walkthrough, see [Understanding spec.yaml](./understanding-spec.md).

---

## Tasks — The Building Blocks

A task is a Python function decorated with `@register()`:

```python
from wt_registry import register
from typing import Annotated
from pydantic import Field

@register()
def apply_color_map(
    df: Annotated[AnyDataFrame, Field(description="Input dataframe", exclude=True)],
    input_column_name: Annotated[str, Field(description="Column to map colors to")],
    colormap: Annotated[str, Field(description="Matplotlib colormap name")] = "tab20b",
    output_column_name: Annotated[str, Field(description="Name for the color column")] = "colormap",
) -> Annotated[AnyDataFrame, Field()]:
    ...
```

The type annotations serve double duty:

1. **Runtime validation** — Pydantic enforces types when the task is called.
2. **Form generation** — The compiler translates `Annotated[type, Field(...)]` into JSON Schema, which Desktop renders as configuration form fields.

Special annotations:

- **`AdvancedField`** — Hides a parameter behind an "Advanced" toggle in the Desktop form.
- **Connection protocol types** (`EarthRangerClient`, `SmartClient`, `EarthEngineClient`) — Tell Desktop to present a data-source picker instead of a text field.
- **Pandera schemas** (`DataFrame[EventGDFSchema]`) — Validate DataFrame structure at runtime.

The Platform SDK ships ~80 built-in tasks — see [Built-in Tasks](./built-in-tasks.md). To write your own, follow [Your First Custom Task](./tutorials/first-custom-task.md).

---

## The Dashboard Model

Every workflow ends by assembling a **dashboard** — a collection of widgets arranged in a grid.

The typical pattern for each widget is:

```
compute data  →  render to HTML  →  persist HTML  →  create widget  →  merge views
```

**Widget types**:

| Type | Task | Displays |
|------|------|----------|
| Map | `create_map_widget_single_view` | Interactive EcoMap |
| Plot | `create_plot_widget_single_view` | Plotly chart (bar, line, pie, time-series) |
| Table | `create_table_widget_single_view` | HTML table |
| Single Value | `create_single_value_widget_single_view` | A single number (count, mean, etc.) |
| Text | `create_text_widget_single_view` | Free-form text |

**`gather_dashboard`** collects all widgets, the time range, and grouper definitions into a final `Dashboard` object that Desktop renders.

**`layout.json`** controls grid positioning. The dashboard uses a 10-column grid. Each entry maps a `widget_id` (index into the `gather_dashboard.widgets` array) to a position (`x`, `y`) and size (`w`, `h`). See the [Widgets tutorial](./tutorials/widgets.md) for details.

**Groupers** let you slice data into multiple views — for example, by month or by species. The dashboard shows a dropdown so the user can switch between views. When groupers are active, each widget is created once per group via `mapvalues`, then merged with `merge_widget_views`. See the [Groupers tutorial](./tutorials/groupers.md) for a full walkthrough.
