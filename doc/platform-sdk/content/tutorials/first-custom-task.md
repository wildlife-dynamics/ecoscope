# Your First Custom Task

All the tasks you have seen so far are built-in — they ship with the `ecoscope-platform` package. In this tutorial you will write your own task, add it to a workflow spec, and run it in Desktop.

---

## What is `@register()`?

Every task in a wt workflow is a plain Python function decorated with `@register()` from the `wt-registry` package. The decorator does two things:

1. Registers the function by name so `spec.yaml` can reference it.
2. Generates JSON Schema from the function's type annotations so Desktop can build a configuration form.

```python
from wt_registry import register

@register()
def my_task(x: int, y: int) -> int:
    return x + y
```

For the full decorator API, see the [wt-registry Reference](https://wt.readthedocs.io/en/latest/reference/wt-registry/).

---

## Step 1 — Create a local task package

Use `uv` to scaffold a Python package:

```console
$ uv init --lib my-tasks
$ cd my-tasks
```

Edit `pyproject.toml` to add the `wt-registry` dependency and the entry point that wt uses to discover your tasks:

```toml
[project]
name = "my-tasks"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["wt-registry"]

[project.entry-points."wt.tasks"]
my_tasks = "my_tasks"
```

The entry point key (`my_tasks`) can be any name. The value (`"my_tasks"`) is the Python module that wt will import to discover `@register()` functions.

---

## Step 2 — Write a task function

Create `src/my_tasks/__init__.py`:

```python
from typing import Annotated
from wt_registry import register
from pydantic import Field

@register()
def add_description_column(
    df: Annotated["pandas.DataFrame", Field(description="Input dataframe", exclude=True)],
    column_name: Annotated[str, Field(description="Name of the new column")] = "description",
    value: Annotated[str, Field(description="Value to fill the column with")] = "No description",
) -> Annotated["pandas.DataFrame", Field()]:
    """Adds a new column with a fixed string value to every row."""
    import pandas

    df = df.copy()
    df[column_name] = value
    return df
```

Key points:

- **`Annotated[type, Field(...)]`** — Pydantic's `Field` provides the description, default value, and JSON Schema metadata. Desktop uses this to render form fields.
- **`exclude=True`** — Marks a parameter as *not* user-configurable. It will only be settable via `partial` in the spec (typically wired from an upstream task).
- **Return type** — Must also be annotated. The compiler uses it to validate downstream references.

---

## Step 3 — Add the task to spec.yaml

Back in your workflow directory, open `spec.yaml` and make two changes:

### Declare your package as a requirement

Add your local package to the `requirements` block using `path:` for local development:

```yaml
requirements:
  - name: "ecoscope-platform"
    version: ">=2.11.3,<3"
    channel: "https://repo.prefix.dev/ecoscope-workflows/"
  - name: "my-tasks"
    path: "../my-tasks"
    editable: true
```

The `editable: true` flag means you can change your task code and recompile without reinstalling the package.

### Insert your task into the workflow

Add a new task instance after `get_events_data` and before `events_colormap`. Wire it into the DAG:

```yaml
  - name: Add Description
    id: add_description
    task: add_description_column
    partial:
      df: ${{ workflow.get_events_data.return }}
      column_name: description
      value: "Event from EarthRanger"

  - name: Events Colormap
    id: events_colormap
    task: apply_color_map
    partial:
      df: ${{ workflow.add_description.return }}   # ← changed from get_events_data
      input_column_name: event_type
      colormap: tab20b
      output_column_name: event_type_colormap
```

---

## Step 4 — Recompile and run

```console
$ wt-compiler compile --spec=spec.yaml --pkg-name-prefix=ecoscope-workflows --results-env-var=ECOSCOPE_WORKFLOWS_RESULTS --install
```

Load the updated template in Desktop, run it, and verify that your new column appears in the data. You have now written, registered, and executed a custom task.

---

## Next steps

- **[Custom Task Patterns](./custom-task-patterns.md)** — Use Platform SDK types for richer integration with Desktop.
- **[Built-in Tasks](../built-in-tasks.md)** — Check if a built-in task already does what you need before writing your own.
