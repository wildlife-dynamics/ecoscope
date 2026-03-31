# Concepts

This page introduces the core ideas behind the Ecoscope Platform SDK. Read it once to build a mental model, then move on to [Getting Started](./getting-started.md) for hands-on work.

---

## The Ecoscope Ecosystem

Three layers work together:

**ecoscope (core library)** — Geospatial analysis primitives: relocations, trajectories, time-density, EcoMap, EcoPlot. If you need a low-level algorithm, it probably lives here.

**Ecoscope Platform SDK (`ecoscope.platform`)** — A curated set of typed *tasks*, *schemas*, *connections*, *groupers*, and *widgets* built on top of the core library. The Platform SDK is what you import when writing workflow specs. It ships ~80 ready-to-use tasks (see [Built-in Tasks](./built-in-tasks.md)).

**Workflow Toolkit (wt)** — A generic, framework-level toolkit for declaring, compiling, and running workflows. wt knows nothing about ecology — it provides `spec.yaml` syntax, the `@register()` decorator, the compiler, and runtime operators like `map`, `mapvalues`, and `skipif`. The Platform SDK plugs into wt. For deep dives into the framework, see the [wt documentation](https://wt.readthedocs.io/en/latest/concepts/).

**Ecoscope Desktop / Ecoscope Web** — End-user applications that load compiled workflow templates, present configuration forms, execute workflows, and display dashboards. You write the spec; either Desktop or Web can run it. Users of these docs will primarily use Ecoscope Desktop for development, as custom workflows are not yet supported for general contribution on Ecoscope Web.

---

## The Development Loop

The core cycle looks like this:

```
edit spec.yaml  →  compile  →  load into Desktop  →  run  →  observe  →  iterate
```

**`@register()` your Python functions** — Before referencing tasks in `spec.yaml`, they must be registered using the `@register()` decorator from `wt-registry`. The Platform SDK's built-in tasks are pre-registered. For custom tasks, see [Your First Custom Task](./tutorials/first-custom-task.md).

**`spec.yaml`** is the single source of truth for your workflow. It declares which tasks to run, how they connect, and which parameters are exposed to the user.

**`wt-compiler compile`** reads `spec.yaml` and produces a standalone, runnable workflow artifact — a pixi workspace complete with a `pixi.toml`, generated Python entry point, and a configuration form schema that Ecoscope Desktop / Ecoscope Web renders for end users.

Two companion files sit alongside `spec.yaml`:

- **`layout.json`** — Controls the dashboard grid layout (widget sizes and positions). This applies to **Dashboard workflows**, which emit an interactive dashboard. There are also **OutputFiles workflows**, which generate files for the user to download without a visual dashboard component.
- **`test-cases.yaml`** — Supplies test/fixture inputs so you can run the workflow locally without a live data source.

The [Getting Started](./getting-started.md) guide walks you through this cycle end-to-end.

---

## Anatomy of a Workflow

A workflow is a **directed acyclic graph (DAG)** of tasks, declared in the `workflow` list inside `spec.yaml`. Each entry is a *task instance*:

```yaml
- name: Events Colormap        # Display name in the configuration form
  id: events_colormap           # Unique identifier for references
  task: apply_color_map         # The registered task function to call
  partial:                      # Arguments bound at compile time
    df: ${{ workflow.get_events_data.return }}
    colormap: tab20b
```

Key ideas:

- **`id`** — A unique identifier. Other tasks reference this task's output via `${{ workflow.<id>.return }}`.
- **`task`** — The name of a Python function decorated with `@register()`. This is how `spec.yaml` maps to code. The registered functions used in your workflow must be contained within packages listed in the `requirements:` section of `spec.yaml`.
- **`partial`** — Values you fix at compile time. Anything listed under `partial` is *not* exposed as a form field in the configuration form displayed to end users in Ecoscope Desktop / Ecoscope Web. Anything *not* listed becomes a configurable form field. This is how you control what parameters are presented to the user in the configuration form.
- **`${{ workflow.<id>.return }}`** — An expression that wires one task's output into another task's input, forming the edges of the DAG.

!!! note
    This is not an exhaustive summary of all `spec.yaml` syntax. For the full reference, see the [wt spec.yaml documentation](https://wt.readthedocs.io/en/latest/reference/spec-yaml/).

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

1. **Runtime validation** — Pydantic handles deserialization and coercion from webform inputs.
2. **Form generation** — The compiler translates type hints into JSON Schema. `Annotated` with `Field(...)` can be used to customize the form generation, but is not required — any type annotation will be converted. Ecoscope Desktop / Ecoscope Web renders this as a configuration form.

The type hint → JSON Schema generation is very flexible, allowing for customization of form behavior. For details, see [Form Customization](./tutorials/form-customization.md). Some highlights:

- **`AdvancedField`** — Hides a parameter behind an "Advanced" toggle in the configuration form in Ecoscope Desktop / Ecoscope Web. Import from `ecoscope.platform.annotations`. See [Form Customization](./tutorials/form-customization.md#advancedfield).
- **Connection protocol types** (`EarthRangerConnection`, `SMARTConnection`, `GoogleEarthEngineConnection`) — Tell Ecoscope Desktop / Ecoscope Web to present a data-source picker instead of a text field.
- **Pandera schemas** (e.g., `DataFrame[EventGDFSchema]`) — Validate DataFrame structure at runtime.

The Platform SDK ships ~80 built-in tasks — see [Built-in Tasks](./built-in-tasks.md). To write your own, follow [Your First Custom Task](./tutorials/first-custom-task.md).

---

## The Dashboard Model

Workflows can produce either a **dashboard** (a collection of widgets arranged in a grid) or **output files** (a directory of files for download). Dashboard workflows are the most common. Here is how the dashboard model works:

**Widget types**:

| Type | Task | Displays |
|------|------|----------|
| Map | [`create_map_widget_single_view`][ecoscope.platform.tasks.results.create_map_widget_single_view] | Interactive EcoMap |
| Plot | [`create_plot_widget_single_view`][ecoscope.platform.tasks.results.create_plot_widget_single_view] | Plotly chart (bar, line, pie, time-series) |
| Table | [`create_table_widget_single_view`][ecoscope.platform.tasks.results.create_table_widget_single_view] | HTML table |
| Single Value | [`create_single_value_widget_single_view`][ecoscope.platform.tasks.results.create_single_value_widget_single_view] | A single number (count, mean, etc.) |
| Text | [`create_text_widget_single_view`][ecoscope.platform.tasks.results.create_text_widget_single_view] | Free-form text |

[`gather_dashboard`][ecoscope.platform.tasks.results.gather_dashboard] collects all widgets, the time range, and grouper definitions into a final `Dashboard` object that Ecoscope Desktop / Ecoscope Web renders.

**`layout.json`** controls grid positioning. See the [Widgets tutorial](./tutorials/widgets.md) for details on the 10-column grid layout system.

**Groupers** let you slice data into multiple views — for example, by month or by species. The dashboard shows a dropdown so the user can switch between views. See the [Groupers tutorial](./tutorials/groupers.md) for a full walkthrough.
