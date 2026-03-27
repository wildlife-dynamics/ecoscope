# Form Customization

The Desktop configuration form is generated from your task function signatures and `spec.yaml` bindings. This tutorial covers how to control which fields appear and how they are presented.

Cross-references: [jsonschema reference](../reference/jsonschema.md), [annotations reference](../reference/annotations.md)

---

## How `partial` controls form fields

The fundamental rule: **bound parameters are fixed; unbound parameters become form fields**.

If a parameter is listed under `partial` in `spec.yaml`, it is bound at compile time and does *not* appear in the Desktop form. Everything else becomes a configurable field.

```yaml
- id: time_range
  task: set_time_range
  partial:
    time_format: '%d %b %Y %H:%M:%S'   # fixed — not in the form
    # start and end are NOT listed → they appear as form fields
```

To expose more parameters, remove them from `partial`. To hide parameters, add them.

---

## AdvancedField

`AdvancedField` hides a parameter behind an "Advanced" toggle in Desktop. The parameter is still configurable, but only visible to users who expand the advanced section:

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

Use `AdvancedField` for parameters that have sensible defaults and are rarely changed — style overrides, debug flags, format strings, etc.

---

## `json_schema_extra`

Pydantic's `Field` accepts `json_schema_extra` to add arbitrary keys to the generated JSON Schema. Desktop respects standard JSON Schema properties like `title`, `description`, and `default`:

```python
@register()
def my_task(
    name: Annotated[str, Field(
        description="Display name",
        json_schema_extra={"title": "Workflow Name", "default": "Untitled"},
    )],
) -> ...:
```

This is useful for overriding the auto-generated title (which defaults to the parameter name) or providing a different default for the form.

---

## `rjsf-overrides`

For customizations that go beyond what type annotations can express, `spec.yaml` supports `rjsf-overrides` at the task-instance level. These overrides modify the JSON Schema and UI Schema that Desktop uses to render the form.

Common use cases:

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

### Custom defaults and constraints

```yaml
- id: my_task
  task: my_custom_task
  rjsf-overrides:
    schema:
      properties:
        threshold:
          default: 0.5
          minimum: 0
          maximum: 1
```

---

## Validators and constraints

Pydantic validators in your task function affect form behavior. For example, `ge=0` (greater than or equal to zero) translates to `minimum: 0` in JSON Schema, which Desktop enforces in the form:

```python
@register()
def set_threshold(
    value: Annotated[float, Field(ge=0, le=1, description="Threshold between 0 and 1")],
) -> float:
    return value
```

Enum types render as dropdowns:

```python
from enum import Enum

class ColorScheme(str, Enum):
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"

@register()
def set_colors(
    scheme: Annotated[ColorScheme, Field(description="Color scheme")] = ColorScheme.VIRIDIS,
) -> str:
    return scheme.value
```

---

## Next steps

- **[Built-in Tasks](../built-in-tasks.md)** — Browse available tasks and find the right one for your workflow.
- **[jsonschema reference](../reference/jsonschema.md)** — Full API for JSON Schema form utilities.
- **[annotations reference](../reference/annotations.md)** — Full API for `AdvancedField` and DataFrame annotations.
