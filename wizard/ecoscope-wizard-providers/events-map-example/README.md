# ecoscope-events-map-example

An example wizard provider that scaffolds an [Ecoscope](https://github.com/wildlife-dynamics/ecoscope) events map workflow project via [wt-compiler](../wt-compiler/).

## Overview

This package registers an `EcoscopeEventsMapExampleProvider` under the `wt_compiler.wizard_providers` entry point group. It extends `DefaultWizardProvider` with one additional question — the events map display title — and omits the requirements loop (the requirement on `ecoscope-platform` is fixed in the template).

### What it generates

| File | Description |
|------|-------------|
| `spec.yaml` | Complete workflow spec: fetches events from EarthRanger, applies a colormap by event type, renders an ecomap, and assembles a single-widget dashboard |
| `layout.json` | Single full-width map widget layout |

### Questions asked

| Field | Description | Default |
|-------|-------------|---------|
| `workflow_id` | Python identifier for the workflow | — |
| `workflow_name` | Human-readable workflow name | — |
| `workflow_description` | Optional description | `""` |
| `author_name` | Author name | — |
| `license_type` | License (`BSD-3-Clause`, `MIT`, `Apache-2.0`) | `BSD-3-Clause` |
| `events_map_title` | Display title for the events map widget (max 50 chars) | `"Events Map"` |

The `requirements` question is skipped — the template hardcodes a dependency on `ecoscope-platform>=2.11.3,<3` from the `ecoscope-workflows` prefix.dev channel.

## Installation

```bash
pip install ecoscope-wizard-providers
```

Or with pixi (from the `wizard/ecoscope-wizard-providers` directory):

```bash
pixi add ecoscope-wizard-providers
```

## Usage

### Interactive wizard

```bash
wt-compiler scaffold run ecoscope-events-map-example --outdir my-events-map/
```

The wizard prompts for workflow ID, name, description, author, license, and the events map title.

### Batch / non-interactive mode

```bash
wt-compiler scaffold run ecoscope-events-map-example \
  --workflow-id my_events_map \
  --workflow-name "My Events Map" \
  --workflow-description "Events overview" \
  --author-name "Wildlife Dynamics" \
  --license-type MIT \
  --events-map-title "Events Map" \
  --outdir my-events-map/
```

## Development

From `wizard/ecoscope-wizard-providers/`:

```bash
uv run pytest
uv run mypy events-map-example/src/ecoscope_events_map_example
uv run ruff check events-map-example/src/ecoscope_events_map_example
```
