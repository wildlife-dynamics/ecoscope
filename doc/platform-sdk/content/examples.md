# Examples

The best way to learn beyond the tutorials is to study production workflows. The [ecoscope-platform-workflows-releases](https://github.com/ecoscope-platform-workflows-releases) GitHub organization hosts open-source workflow repositories that run in production.

---

## Production workflows

| Workflow | Widgets | Use Case | Key Patterns |
|----------|---------|----------|--------------|
| [events](https://github.com/ecoscope-platform-workflows-releases/events) | 4 | Event occurrences from EarthRanger | Groupers, bar/pie/map widgets |
| [event-details](https://github.com/ecoscope-platform-workflows-releases/event-details) | 7 | Event detail field analysis | Conditional skipif, summary tables, single-value widgets |
| [subject-tracking](https://github.com/ecoscope-platform-workflows-releases/subject-tracking) | 10 | Trajectory analysis for tracked subjects | Relocations to trajectories, time density, speed profiles |
| [patrols](https://github.com/ecoscope-platform-workflows-releases/patrols) | 9 | Patrol effort and embedded events | Dual data streams, dynamic categorization, rjsf-overrides |

---

## What to look for in each repository

Every workflow repository follows the same structure:

- **`spec.yaml`** — The workflow definition. Start here to understand the DAG, task wiring, and `partial` bindings.
- **`layout.json`** — Dashboard grid layout. See how widgets are positioned and sized.
- **`test-cases.yaml`** — Test/fixture inputs for offline testing. Shows which parameters are required and what the test data format looks like.
- **`__results_snapshots__/`** — Snapshotted outputs from test runs that prevent visual regression. Useful for understanding what each task produces.

---

## Suggested learning path

1. **Start with [events](https://github.com/ecoscope-platform-workflows-releases/events)** — The simplest multi-widget workflow. It extends the intro example from [Getting Started](./getting-started.md) with [groupers](./tutorials/groupers.md), a bar chart, and a pie chart.

2. **Then [event-details](https://github.com/ecoscope-platform-workflows-releases/event-details)** — Introduces conditional skip logic, summary tables, and single-value widgets. Shows how to handle workflows where some tasks may not produce output.

3. **Then [subject-tracking](https://github.com/ecoscope-platform-workflows-releases/subject-tracking) or [patrols](https://github.com/ecoscope-platform-workflows-releases/patrols)** — These are the most complex workflows, with multiple data streams, preprocessing pipelines (relocations to trajectories), time density analysis, and advanced form customization.

These workflows are EarthRanger-specific and will not work with other data sources without modification.

---

## Using an example as a starting point

To use a production workflow as a starting point for your own:

1. Clone the repository.
2. Study `spec.yaml` to understand the pipeline.
3. Copy and modify — add your custom tasks, change the data source, or swap out widgets.
4. Compile and test locally, or import directly into Ecoscope Desktop for interactive testing.

For writing custom tasks, see [Your First Custom Task](./tutorials/first-custom-task.md).
