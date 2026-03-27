# Data Sources

Workflows load data from external systems through **connections**. Each connection type wraps a client library and provides a consistent interface for Desktop.

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

Desktop renders a data-source picker for `set_er_connection` because the task's parameter type is a connection protocol.

At runtime, connection credentials are resolved from **environment variables** that Desktop sets based on the user's configured data sources.

---

## EarthRanger

**Connection task**: `set_er_connection`

**Available data-loading tasks**:

- `get_events` — Load events with optional filtering by type, category, and time range
- `get_patrol_events` — Load events embedded in patrols
- `get_patrols` — Load patrol tracks
- `get_patrol_observations` — Load observations from patrols
- `get_subjectgroup_observations` — Load GPS observations for a subject group

**Environment variables** (set by Desktop):

- `ECOSCOPE_ER_SERVER` — EarthRanger site URL
- `ECOSCOPE_ER_USERNAME` — API username
- `ECOSCOPE_ER_PASSWORD` — API password (or token)

**Typical spec pattern**:

```yaml
- id: er_client_name
  task: set_er_connection

- id: time_range
  task: set_time_range

- id: get_events_data
  task: get_events
  partial:
    client: ${{ workflow.er_client_name.return }}
    time_range: ${{ workflow.time_range.return }}
    event_columns:
      - id
      - time
      - event_type
      - geometry
    raise_on_empty: false
```

---

## SMART

**Connection task**: `set_smart_connection`

**Available data-loading tasks**:

- `get_events_from_smart` — Load events from a SMART Connect server
- `get_patrol_observations_from_smart` — Load patrol observations

**Environment variables**:

- `ECOSCOPE_SMART_SERVER` — SMART Connect URL
- `ECOSCOPE_SMART_USERNAME` — API username
- `ECOSCOPE_SMART_PASSWORD` — API password

---

## Earth Engine

**Connection task**: `set_gee_connection`

**Available data-loading tasks**:

- `download_roi` — Download a region of interest from Google Earth Engine
- `calculate_ndvi_range` — Calculate NDVI statistics for a region

**Environment variables**:

- `ECOSCOPE_GEE_SERVICE_ACCOUNT` — GEE service account email
- `ECOSCOPE_GEE_SERVICE_ACCOUNT_KEY` — Path to service account key file

---

## Named connections and testing

The `DataConnection.from_named_connection()` factory method resolves connection parameters from environment variables by name. This is how Desktop maps user-configured data sources to runtime credentials.

In `test-cases.yaml`, you can bypass connections entirely by mocking the data-loading tasks:

```yaml
- test_name: "offline"
  mock_tasks:
    get_events:
      return:
        loader: "parquet"
        path: "tests/sample_events.geoparquet"
```

This lets you develop and test workflows without access to a live EarthRanger, SMART, or GEE instance.

---

## Next steps

- **[Widgets](./widgets.md)** — Visualize the data you loaded.
- **[connections reference](../reference/connections.md)** — Full API for connection classes and protocols.
