# Troubleshooting

This page covers common errors and how to resolve them.

---

## Reading error messages

When a task fails at runtime, wt produces a **`TaskInstanceError`** that identifies which task failed and why:

```
TaskInstanceError: Task 'get_events_data' (get_events) failed:
  ConnectionError: Unable to connect to EarthRanger at https://example.pamdas.org
```

The format is: `Task '<id>' (<task function>) failed: <exception>`. The `id` matches your `spec.yaml`, so you can quickly find the relevant task instance.

---

## "Why is my widget empty?"

Empty widgets are usually caused by the **skip system**. When a task is skipped, it returns a `SkipSentinel` instead of a normal value. If the default `task-instance-defaults` includes `any_dependency_skipped`, downstream tasks will also be skipped in a chain reaction.

**Diagnosis steps**:

1. Check the workflow execution log for "skipped" messages.
2. Find the *first* task that was skipped — that is the root cause.
3. Common triggers:
    - `any_is_empty_df` — The DataFrame input was empty (no data in the time range, wrong event types, etc.).
    - `all_geometry_are_none` — The data has no geometry column, or all geometry values are null.

**Widget tasks should use `skipif: never`** so they always produce a result (possibly with empty data) rather than being skipped entirely. This ensures the dashboard renders correctly. See [Understanding spec.yaml](./understanding-spec.md#events_map_widget--skipif-never).

---

## Common compilation errors

### Missing type annotations

```
CompilationError: Function 'my_task' parameter 'x' has no type annotation
```

Every parameter and the return type must have a type annotation. The compiler uses these to generate JSON Schema for the Desktop form.

**Fix**: Add type annotations to all parameters:

```python
# Before (error)
@register()
def my_task(x, y):
    return x + y

# After (works)
@register()
def my_task(x: int, y: int) -> int:
    return x + y
```

### Duplicate task registrations

```
RegistrationError: Task 'my_task' is already registered
```

Two functions with the same name are decorated with `@register()`. Task names must be globally unique across all packages in the workflow's requirements.

**Fix**: Rename one of the functions, or use the `name` parameter on `@register()` to give it a distinct registration name.

### Schema generation failures

```
SchemaGenerationError: Unable to generate JSON Schema for type 'MyCustomClass'
```

The compiler cannot convert a parameter's type to JSON Schema. This happens with custom classes that Pydantic does not know how to serialize.

**Fix**: Use standard types (`str`, `int`, `float`, `bool`, `list`, `dict`), Pydantic models, or Platform SDK types (`AnyDataFrame`, `AnyGeoDataFrame`). See [Custom Task Patterns](./tutorials/custom-task-patterns.md).

---

## Empty DataFrames

If your workflow produces no data:

1. **Check the time range** — Is there data in the selected period? Try widening the range.
2. **Check the event types / categories** — Are you filtering for types that exist in the data source?
3. **Check the connection** — Can Desktop connect to EarthRanger / SMART / GEE? Look for connection errors in the log.
4. **Check `raise_on_empty`** — If set to `false`, the task silently returns an empty DataFrame. Set to `true` during debugging to get an explicit error.

---

## Testing locally

### Calling tasks in a REPL

`@register()` functions are plain Python functions — you can import and call them directly:

```python
from ecoscope.platform.tasks.transformation import apply_color_map
import pandas as pd

df = pd.DataFrame({"event_type": ["fire", "poaching", "fire"]})
result = apply_color_map(df=df, input_column_name="event_type", colormap="tab20b")
print(result.columns)
```

### Using `test-cases.yaml`

Define test cases with mock data to run the full workflow offline:

```yaml
- test_name: "basic"
  mock_tasks:
    get_events:
      return:
        loader: "parquet"
        path: "tests/sample_events.geoparquet"
```

Run with:

```console
$ cd <compiled-workflow-dir>
$ pixi run test
```

Failed tests will show which task produced unexpected output, with a diff against the expected snapshot in `__results_snapshots__/`.

---

## Getting help

If you are stuck:

1. Check the [wt documentation](https://wt.readthedocs.io/) for framework-level issues (spec syntax, compilation, runtime).
2. Check the [Reference](./reference/tasks/io.md) pages for task-specific parameter documentation.
3. Study the [Examples](./examples.md) for working patterns you can adapt.
