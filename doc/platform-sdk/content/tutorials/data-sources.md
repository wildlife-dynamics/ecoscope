# Data Sources

Workflows load data from external systems through **connections**. Each connection type wraps a client library and provides a consistent interface for Ecoscope Desktop / Ecoscope Web.

Cross-references: [connections reference](../reference/connections.md), [io tasks reference](../reference/tasks/io.md)

---

## How connections work

A connection is a typed wrapper around an API client. In `spec.yaml`, you use a `set_*_connection` task to create the connection, then pass it to data-loading tasks via `${{ workflow.<id>.return }}`:

```yaml
- name: Data Source
  id: er_client_name
  task: set_er_connection

- name: Get Events
  id: get_events_data
  task: get_events
  partial:
    client: ${{ workflow.er_client_name.return }}
    time_range: ${{ workflow.time_range.return }}
    ...
```

Ecoscope Desktop / Ecoscope Web renders a data-source picker for `set_er_connection` because the task's parameter type is a connection protocol.

At runtime, connection credentials are resolved from **environment variables** that Ecoscope Desktop / Ecoscope Web sets based on the user's configured data sources.

---

## EarthRanger

**Connection task**: [`set_er_connection`][ecoscope.platform.tasks.io.set_er_connection]

For available data-loading tasks and their parameters, see the [io tasks reference](../reference/tasks/io.md).

---

## SMART

**Connection task**: [`set_smart_connection`][ecoscope.platform.tasks.io.set_smart_connection]

For available data-loading tasks and their parameters, see the [io tasks reference](../reference/tasks/io.md).

---

## Earth Engine

**Connection task**: [`set_gee_connection`][ecoscope.platform.tasks.io.set_gee_connection]

For available data-loading tasks and their parameters, see the [io tasks reference](../reference/tasks/io.md).

---

## Next steps

- **[Widgets](./widgets.md)** — Visualize the data you loaded.
- **[connections reference](../reference/connections.md)** — Full API for connection classes and protocols.
