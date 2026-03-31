# Tutorials

These tutorials build on [Getting Started](./getting-started.md) and [Understanding spec.yaml](./understanding-spec.md). Work through them in order — each one introduces concepts used by the next.

---

1. **[Your First Custom Task](./tutorials/first-custom-task.md)** — Write a Python function, register it with `@register()`, add it to your spec, and run it in Ecoscope Desktop.

2. **[Data Sources](./tutorials/data-sources.md)** — Connect to EarthRanger, SMART, and Earth Engine, and understand the connection/environment-variable pattern.

3. **[Widgets](./tutorials/widgets.md)** — Build map, plot, table, single-value, and text widgets, assemble them into a dashboard, and control layout with `layout.json`.

4. **[Groupers](./tutorials/groupers.md)** — Slice data into views with `AllGrouper`, `ValueGrouper`, `TemporalGrouper`, and `SpatialGrouper`, and understand `map` vs `mapvalues` operators.

5. **[Form Customization](./tutorials/form-customization.md)** — Control which parameters appear in the Ecoscope Desktop / Ecoscope Web configuration form, hide fields behind "Advanced", and override JSON Schema with `rjsf-overrides`.

---

The tutorials use minimal spec.yaml snippets that extend the intro example from [Getting Started](./getting-started.md). For full production workflows, see [Examples](./examples.md).
