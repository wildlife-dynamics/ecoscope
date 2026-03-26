# ecoscope-events-dashboard-wizard-provider

Wizard provider for scaffolding [Ecoscope](https://github.com/wildlife-dynamics/ecoscope) events dashboard workflow projects via [wt-compiler](../wt-compiler/).

## Overview

This package extends `DefaultWizardProvider` with an interactive widget configuration loop.  It generates:

- **`spec.yaml`** — complete events dashboard workflow spec with conditional task chains for each enabled widget type
- **`layout.json`** — auto-generated 2-column dashboard grid layout based on the selected widgets and their order

### Available widget types

| Widget | Description |
|--------|-------------|
| `bar_chart` | Events over time, grouped by type |
| `events_map` | Point map of event locations |
| `pie_chart` | Event type distribution |
| `event_count_map` | Grid heatmap of event density |
| `events_table` | Filterable/sortable event table |

Widget insertion order determines `widget_id` assignment in `layout.json` and the `gather_dashboard.widgets` list order.

## Installation

```bash
pip install ecoscope-events-dashboard-wizard-provider
```

Or with pixi (from the monorepo):

```bash
pixi add ecoscope-events-dashboard-wizard-provider
```

## Usage

### Interactive wizard

```bash
wt-compiler scaffold run ecoscope-events-dashboard --outdir my-dashboard/
```

The wizard will prompt for standard workflow fields (ID, name, description, author, license, requirements) followed by a widget configuration loop where you select widget types and display titles.

### Batch / non-interactive mode

```bash
wt-compiler scaffold run ecoscope-events-dashboard \
  --workflow-id my_events_dashboard \
  --workflow-name "My Events Dashboard" \
  --workflow-description "Events overview" \
  --author-name "Wildlife Dynamics" \
  --license-type MIT \
  --widgets '{"widget": "events_map", "title": "Events Map"}' \
  --widgets '{"widget": "bar_chart", "title": "Events Over Time"}' \
  --outdir my-dashboard/
```

Each `--widgets` flag accepts a JSON object with `widget` (one of the types above) and `title` (max 50 characters).

## Development

```bash
cd ecoscope-events-dashboard-wizard-provider
uv run pytest
uv run mypy src/ecoscope_events_dashboard_wizard_provider
uv run ruff check src/ecoscope_events_dashboard_wizard_provider
```
