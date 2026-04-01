# Form Customization

The Ecoscope Desktop / Ecoscope Web configuration form is generated from your task function signatures and `spec.yaml` bindings. This tutorial covers how to control which fields appear and how they are presented.

Cross-references: [annotations reference](../reference/annotations.md)

---

## How `partial` controls form fields

The fundamental rule: **bound parameters are fixed; unbound parameters become form fields**.

If a parameter is listed under `partial` in `spec.yaml`, it is bound at compile time and does *not* appear in the Ecoscope Desktop / Ecoscope Web configuration form. Everything else becomes a configurable field.

!!! note
    Parameters specified as `map` or `mapvalues` arguments in the spec.yaml are also excluded from the form. See the [Groupers tutorial](./groupers.md) for details.

```yaml
- id: time_range
  task: set_time_range
  partial:
    time_format: '%d %b %Y %H:%M:%S'   # fixed — not in the form
    # start and end are NOT listed → they appear as form fields
```

To expose more parameters, remove them from `partial`. To hide parameters, add them.

See the [full signature of `set_time_range`][ecoscope.platform.tasks.filter.set_time_range] in the Reference to understand all available parameters.

---

## `Field` — Standard parameter customization

Pydantic's `Field` is the standard way to customize how a parameter appears in the configuration form. Common arguments:

- **`title`** — Override the auto-generated field title (which defaults to the parameter name)
- **`description`** — Help text displayed in the form
- **`default`** — Default value
- **`ge`, `le`, `gt`, `lt`** — Numeric constraints (translated to JSON Schema `minimum`/`maximum`)
- **`exclude=True`** — Marks a parameter as not user-configurable (settable only via `partial`)

```python
from pydantic import Field

@register()
def set_threshold(
    value: Annotated[float, Field(
        title="Detection Threshold",
        description="Threshold between 0 and 1",
        ge=0, le=1,
    )] = 0.5,
) -> float:
    return value
```

Numeric constraints like `ge=0` translate to `minimum: 0` in JSON Schema, which Ecoscope Desktop / Ecoscope Web enforces in the form.

---

## AdvancedField

`AdvancedField` hides a parameter behind an "Advanced" toggle in Ecoscope Desktop / Ecoscope Web. The parameter is still configurable, but only visible to users who expand the advanced section:

```python
from ecoscope.platform.annotations import AdvancedField

@register()
def draw_ecomap(
    geo_layers: Annotated[..., Field(exclude=True)],
    tile_layers: Annotated[list[TileLayer] | None, Field(description="Base map layers")] = None,
    static: Annotated[bool, Field(description="Disable map pan/zoom")] = False,
    title: Annotated[str | None, AdvancedField(description="Map title")] = None,        # ← advanced
    max_zoom: Annotated[int, Field(description="Max zoom level")] = 20,
) -> ...:
```

Use `AdvancedField` for parameters that have sensible defaults and are rarely changed — style overrides, debug flags, format strings, etc. `AdvancedField` accepts the same parameters as `Field` (`title`, `description`, `ge`, `le`, etc.).

---

## `json_schema_extra` — Last resort customization

For cases that `Field` and `AdvancedField` cannot handle, Pydantic's `json_schema_extra` adds arbitrary keys to the generated JSON Schema. Use this sparingly — prefer `Field` or `AdvancedField` for standard customizations.

```python
@register()
def my_task(
    name: Annotated[str, Field(
        description="Display name",
        json_schema_extra={"ui:help": "This text appears below the field in the form"},
    )],
) -> ...:
```

---

## `rjsf-overrides`

For customizations that go beyond what type annotations can express, `spec.yaml` supports `rjsf-overrides` at the task-instance level. These overrides modify the JSON Schema and UI Schema that Ecoscope Desktop / Ecoscope Web uses to render the form.

### Dropdown population with `EarthRangerEnumResolver`

When you want a form field to show a dropdown populated from EarthRanger data (e.g., event types):

```yaml
- id: get_events_data
  task: get_events
  rjsf-overrides:
    schema:
      $defs:
        EventTypeEnum:
          resolver: EarthRangerEnumResolver
          params:
            method: get_event_type_choices
    properties:
      event_type:
        $ref: "#/$defs/EventTypeEnum"
```

### Hiding fields in the UI

```yaml
- id: my_task
  task: my_custom_task
  rjsf-overrides:
    uiSchema:
      internal_param:
        "ui:widget": "hidden"
```

---

## Leveraging Pydantic's JSON Schema support

The form generation is built directly on Pydantic's JSON Schema capabilities — we are not reinventing the wheel. Any pattern that Pydantic supports for JSON Schema generation works here, including:

- **Enum types** — Render as dropdowns automatically
- **`WithJsonSchema`** — Override the generated schema for a specific type
- **`SkipJsonSchema`** — Exclude a type from schema generation
- **Custom `__get_pydantic_json_schema__`** — Full control over schema output

For the complete reference, see the [Pydantic JSON Schema documentation](https://docs.pydantic.dev/latest/concepts/json_schema/).

---

## Next steps

- **[Built-in Tasks](../built-in-tasks.md)** — Browse available tasks and find the right one for your workflow.
- **[annotations reference](../reference/annotations.md)** — Full API for `AdvancedField` and DataFrame annotations.
