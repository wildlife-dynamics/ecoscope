# Troubleshooting

This page covers common errors and how to resolve them.

---

## Reading error messages

When a task fails at runtime, the workflow produces a **`TaskInstanceError`** that identifies which task failed and why:

```
TaskInstanceError: Task 'get_events_data' (get_events) failed:
  ConnectionError: Unable to connect to EarthRanger at https://example.pamdas.org
```

The format is: `Task '<id>' (<task function>) failed: <exception>`. The `id` matches your `spec.yaml`, so you can quickly find the relevant task instance.

In Ecoscope Desktop / Ecoscope Web, these errors can be found under the **View Metadata** section of the kebab menu for the failed workflow row on the My Workflows table.

---

## "Why is my widget empty?"

Empty widgets (i.e., those that display no data) are usually caused by the **skip system**. When a task is skipped, it returns a `SkipSentinel` instead of a normal value. If the default `task-instance-defaults` includes `any_dependency_skipped`, downstream tasks will also be skipped in a chain reaction.

**Diagnosis steps**:

1. Consider whether the configuration you submitted corresponds to a data selection for which data actually exists. For example, if using EarthRanger data sources, check that events of the selected types exist in the selected time range on your EarthRanger instance.
2. Common triggers:
    - `any_is_empty_df` — The DataFrame input was empty (no data in the time range, wrong event types, etc.).
    - `all_geometry_are_none` — The data has no geometry column, or all geometry values are null.
3. Check the time range — Is there data in the selected period? Try widening the range.
4. Check the event types / categories — Are you filtering for types that exist in the data source?
5. Check the connection — Can Ecoscope Desktop / Ecoscope Web connect to EarthRanger / SMART / GEE? Look for connection errors.

**Widget tasks should use `skipif: never`** so they always produce a result (possibly with empty data) rather than being skipped entirely. This ensures the dashboard renders correctly. See [Understanding spec.yaml](./understanding-spec.md#events_map_widget-skipif-never).

---

## Common compilation errors

### Missing type annotations

```
CompilationError: Function 'my_task' parameter 'x' has no type annotation
```

Every parameter and the return type of all registered Python functions used as tasks in the workflow must have a type annotation.

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

### Schema generation failures

```
SchemaGenerationError: Unable to generate JSON Schema for type 'MyCustomClass'
```

The compiler cannot convert a parameter's type to JSON Schema. This happens with custom classes that Pydantic does not know how to serialize.

**Fix**: Use standard types (`str`, `int`, `float`, `bool`, `list`, `dict`), Pydantic models, or Platform SDK types (`AnyDataFrame`, `AnyGeoDataFrame`). Schema generation for custom objects can be customized with Pydantic. This is covered in the [Form Customization tutorial](./tutorials/form-customization.md).

---

## "Could not read properties of undefined (reading 'name')"

This error appears in Ecoscope Desktop / Ecoscope Web when submitting the configuration form. It means your workflow is missing the required task instance `workflow_details`.

Every ecoscope workflow **must** include:

```yaml
workflow:
  - name: Workflow Details
    id: workflow_details
    task: set_workflow_details
```

**Fix**: Add the missing task(s) to your `spec.yaml`. The `id` values must be exactly `workflow_details` and `time_range` — Ecoscope Desktop and Ecoscope Web look for these specific IDs when processing the configuration form.

---

## Developing against a local ecoscope checkout

When developing or debugging ecoscope itself, you can point your workflow at a local checkout instead of the published `ecoscope-platform` conda package. The conda package name is `ecoscope-platform`, but the PyPI/local package name is `ecoscope` with extras:

```yaml
requirements:
  - name: python
    version: "3.12.*"
  - name: rasterio
    version: "1.5.0"
    channel: conda-forge
  - name: numpy
    version: ">=2,<2.1"
    channel: conda-forge
  - name: ecoscope
    path: /Users/me/ecoscope
    editable: true
    extras: ["platform", "mapping", "analysis"]
```

**Key points:**

- **`python` pin** — without this, the ephemeral discovery environment may
  resolve Python 3.14+, where transitive deps like `pydantic-core` lack
  pre-built wheels.
- **`rasterio` conda dep** — ensures GDAL native libraries are available for
  the editable ecoscope install.
- **`extras`** — the extras you need depend on which tasks your workflow uses and the dependencies.
  The four shown below cover the most common case:
    - `platform`: pandera, pydantic, wt-registry, wt-task (always required)
    - `mapping`: lonboard, matplotlib (results/map tasks)
    - `analysis`: statsmodels (analysis tasks)
    - `plotting`: plotly, scikit-learn

    Other extras may also be needed depending on your tasks/dependencies.
- **`numpy` pin** — ecoscope currently pins `numpy>=2,<2.1`, which conflicts
  with what conda resolves on some platforms. Add a numpy conda requirement to
  the spec to avoid a post-compile patch:

    ```yaml
    - name: numpy
      version: ">=2,<2.1"
      channel: conda-forge
    ```

- **Post-compile pydantic patch** — ecoscope requires `pydantic<2.9.0` (newer
  versions break discriminated unions in `apply_classification`), but the
  compiler emits `pydantic>=2.0.0,<3.0.0` as a conda dependency. After
  compiling, manually edit the generated `pixi.toml` to pin
  `pydantic>=2.0.0,<2.9.0`. This cannot be set via `requirements:` because
  the compiler overrides pydantic's version constraint.

---

## Getting help

If you are stuck:

1. Check the [wt documentation](https://wt.readthedocs.io/) for framework-level issues (spec syntax, compilation, runtime).
2. Check the [Reference](./reference/tasks/io.md) pages for task-specific parameter documentation.
3. Study the [Examples](./examples.md) for working patterns you can adapt.
4. [Open a GitHub Issue](https://github.com/wildlife-dynamics/ecoscope/issues) to report bugs or request features.
