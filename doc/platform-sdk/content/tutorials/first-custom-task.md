# Your First Custom Task

All the tasks you have seen so far are built-in — they ship with the `ecoscope-platform` package. In this tutorial you will write your own task, add it to a workflow spec, and run it in Ecoscope Desktop / Ecoscope Web.

---

## What is `@register()`?

Every task in a wt workflow is a plain Python function decorated with [`@register()`](https://wt.readthedocs.io/en/latest/reference/wt-registry/#register-decorator) from the `wt-registry` package. The decorator does two things:

1. Registers the function by name so `spec.yaml` can reference it.
2. Generates JSON Schema from the function's type annotations so Ecoscope Desktop / Ecoscope Web can build a configuration form.

```python
from wt_registry import register

@register()
def my_task(x: int, y: int) -> int:
    return x + y
```

For the full decorator API, see the [wt-registry Reference](https://wt.readthedocs.io/en/latest/reference/wt-registry/).

---

## Prerequisites

In addition to the tools from [Getting Started](../getting-started.md), you will need [`uv`](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) installed.

---

## Step 1 — Create a local task package

Use `uv` to scaffold a Python package:

```console
$ uv init --lib my-tasks --python 3.11
$ cd my-tasks
```

Edit `pyproject.toml` to add the `wt-registry` dependency and the entry point that `wt-registry` uses to discover your registered functions:

```diff
 [project]
 name = "my-tasks"
 version = "0.1.0"
 requires-python = ">=3.10"
-dependencies = []
+dependencies = ["wt-registry"]
+
+[project.entry-points."wt_registry"]
+my_tasks = "my_tasks"
```

The entry point key (`my_tasks`) can be any name. The value (`"my_tasks"`) is the Python module that wt will import to discover `@register()` functions.

---

### Why write this task?

When you hover over event markers on the map produced by the events-map-example workflow, there are no useful tooltips — just raw data or nothing at all. You want to show a human-readable label like "Fire Alert — 2024-03-15" so users can quickly identify events at a glance. No built-in task does this specific formatting, so you will write a custom one.

---

## Step 2 — Write a task function

Create `src/my_tasks/__init__.py`:

```python
from typing import Annotated
from wt_registry import register
from pydantic import Field
from ecoscope.platform.annotations import AnyGeoDataFrame

@register()
def format_event_labels(
    df: Annotated[AnyGeoDataFrame, Field(description="Input geodataframe", exclude=True)],
    label_column: Annotated[str, Field(description="Name for the new label column")] = "event_label",
) -> Annotated[AnyGeoDataFrame, Field()]:
    """Creates a formatted label by combining event type and date."""
    df = df.copy()
    df[label_column] = df["event_type"] + " — " + df["time"].dt.strftime("%Y-%m-%d")
    return df
```

Key points:

- **`Annotated[type, Field(...)]`** — Pydantic's `Field` provides the description, default value, and JSON Schema metadata. Ecoscope Desktop / Ecoscope Web uses this to render form fields.
- **`exclude=True`** — Marks a parameter as *not* user-configurable. It will only be settable via `partial` in the spec (typically wired from an upstream task).
- **Return type** — Must also be annotated. The compiler uses it to validate downstream references.

---

## Step 3 — Add the task to spec.yaml

Back in your workflow directory, open `spec.yaml` and make two changes:

### Declare your package as a requirement

Add your local package to the `requirements` block using `path:` for local development. **Important**: the `path:` argument must be an **absolute path**, not a relative one:

```yaml
requirements:
  - name: "ecoscope-platform"
    version: ">=2.11.3,<3"
    channel: "https://repo.prefix.dev/ecoscope-workflows/"
  - name: "my-tasks"
    path: "/absolute/path/to/my-tasks"
    editable: true
```

The `editable: true` flag means you can change your task code and recompile without reinstalling the package.

### Insert your task into the workflow

Add `format_event_labels` after `get_events_data`, then update `apply_color_map` to reference the new task's output. Also update `create_point_layer` to include the new label column in tooltips:

```yaml
  - name: Format Event Labels
    id: format_labels
    task: format_event_labels
    partial:
      df: ${{ workflow.get_events_data.return }}
      label_column: event_label

  - name: Events Colormap
    id: events_colormap
    task: apply_color_map
    partial:
      df: ${{ workflow.format_labels.return }}   # ← changed from get_events_data
      input_column_name: event_type
      colormap: tab20b
      output_column_name: event_type_colormap
```

Also update the `create_point_layer` task to show tooltip columns:

```yaml
    tooltip_columns:
      - event_label    # ← your new column
```

---

## Step 4 — Recompile and run

```console
$ wt-compiler compile --spec=spec.yaml --pkg-name-prefix=ecoscope-workflows --results-env-var=ECOSCOPE_WORKFLOWS_RESULTS --install
```

Load the updated template, run it, and hover over the map markers. You should see labels like "Fire Alert — 2024-03-15" in the tooltips. You have now written, registered, and executed a custom task.

---

## Next steps

- **[Data Sources](./data-sources.md)** — Connect to EarthRanger, SMART, and Earth Engine.
- **[Built-in Tasks](../built-in-tasks.md)** — Check if a built-in task already does what you need before writing your own.
