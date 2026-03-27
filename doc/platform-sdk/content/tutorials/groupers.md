# Groupers

Groupers let you slice data into multiple views — for example, by month, by species, or by patrol area. The dashboard shows a dropdown so users can switch between views.

Cross-references: [indexes reference](../reference/indexes.md), [groupby tasks reference](../reference/tasks/groupby.md)

---

## What is a grouper?

A grouper splits a DataFrame into subsets by some criterion. Each subset becomes a separate **view** in the dashboard. The general pattern is:

```
set_groupers  →  split_groups  →  (process each group with mapvalues)  →  merge views
```

`set_groupers` accepts user input for how to group the data. `split_groups` takes a DataFrame and the grouper definition, and returns a **keyed iterable** — a dict-like structure where each key is a composite filter (identifying the group) and each value is the subset of data.

---

## `map` vs `mapvalues` — Two kinds of iteration

wt provides two operators for running a task over a collection. Understanding the difference is essential for grouped workflows.

**`map`** iterates over a **list**, producing a list of results:

```yaml
- id: persist_all
  task: persist_text
  map:
    argnames:
      - text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
    text: ${{ workflow.all_htmls.return }}   # a list
```

**`mapvalues`** iterates over a **dict** (keyed by group), preserving keys:

```yaml
- id: color_per_group
  task: apply_color_map
  mapvalues:
    argnames:
      - df
  partial:
    df: ${{ workflow.split_data.return }}    # a keyed iterable (dict)
    input_column_name: event_type
    colormap: tab20b
```

**The key insight**: after `split_groups`, data is a keyed iterable (dict), so you use **`mapvalues`** for all per-group processing. At the widget creation step, the pipeline transitions to **`map`** because `create_*_widget_single_view` takes `[view, data]` — the view key and the data value as separate arguments.

For a deep dive into these operators, see the [wt Tutorials](https://wt.readthedocs.io/en/latest/tutorials/).

---

## AllGrouper (default)

`AllGrouper` is the simplest grouper — it creates a single view containing all data. This is what the events-map-example uses:

```yaml
- id: groupers
  task: set_groupers
  # No partial → user sees a form, but default is AllGrouper

- id: split_data
  task: split_groups
  partial:
    df: ${{ workflow.get_events_data.return }}
    groupers: ${{ workflow.groupers.return }}
```

With `AllGrouper`, `split_groups` returns a single-entry dict: `{("All", "=", "True"): <full dataframe>}`. The rest of the pipeline processes this one group normally.

---

## ValueGrouper

`ValueGrouper` groups data by the distinct values in a column:

```yaml
- id: groupers
  task: set_groupers
  partial:
    groupers:
      - index_name: event_type
```

This creates one view per unique `event_type` value. The dashboard dropdown lists each event type.

In `test-cases.yaml`, you can specify the grouper:

```yaml
- test_name: "grouped_by_type"
  params:
    set_groupers:
      groupers:
        - index_name: event_type
```

---

## TemporalGrouper

`TemporalGrouper` groups data by time periods. The Platform SDK provides several temporal index types:

| Index | Directive | Example |
|-------|-----------|---------|
| `Year` | `%Y` | 2024 |
| `Month` | `%B` | January |
| `YearMonth` | `%Y-%m` | 2024-01 |
| `Date` | `%Y-%m-%d` | 2024-01-15 |
| `DayOfTheWeek` | `%A` | Monday |
| `DayOfTheMonth` | `%d` | 15 |
| `DayOfTheYear` | `%j` | 015 |
| `Hour` | `%H` | 14 |

```yaml
- id: groupers
  task: set_groupers
  partial:
    groupers:
      - index_name: month
        directive: "%B"
```

---

## SpatialGrouper

`SpatialGrouper` groups data by geographic regions (e.g., conservation areas, patrol zones). It requires a spatial feature group from the data source to define the regions.

---

## The grouped widget pipeline

Here is how a grouped map widget pipeline looks. Compare this with the ungrouped version in the [Widgets tutorial](./widgets.md):

```yaml
# 1. Split data by group
- id: split_data
  task: split_groups
  partial:
    df: ${{ workflow.colored_data.return }}
    groupers: ${{ workflow.groupers.return }}

# 2. Create layers per group (mapvalues — dict in, dict out)
- id: point_layers
  task: create_point_layer
  mapvalues:
    argnames:
      - geodataframe
  partial:
    geodataframe: ${{ workflow.split_data.return }}
    layer_style:
      fill_color_column: color
      get_radius: 5

# 3. Draw maps per group (mapvalues)
- id: ecomaps
  task: draw_ecomap
  mapvalues:
    argnames:
      - geo_layers
  partial:
    geo_layers: ${{ workflow.point_layers.return }}
    tile_layers:
      - layer_name: "TERRAIN"

# 4. Persist HTML per group (mapvalues)
- id: ecomap_htmls
  task: persist_text
  mapvalues:
    argnames:
      - text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
    text: ${{ workflow.ecomaps.return }}

# 5. Create widgets (map — transition from dict to list)
- id: map_widgets
  task: create_map_widget_single_view
  map:
    argnames:
      - view
      - data
  skipif:
    conditions:
      - never
  partial:
    title: "Events Map"
    view: ${{ workflow.ecomap_htmls.return }}
    data: ${{ workflow.ecomap_htmls.return }}

# 6. Merge views into a single grouped widget
- id: map_widget_merged
  task: merge_widget_views
  partial:
    widgets: ${{ workflow.map_widgets.return }}
```

The key transition happens at step 5: `create_map_widget_single_view` uses `map` with `argnames: [view, data]` to unpack the keyed iterable into separate widget views. Then `merge_widget_views` combines them into a single grouped widget with a view dropdown.

In `gather_dashboard`, reference the merged widget:

```yaml
- id: dashboard
  task: gather_dashboard
  partial:
    details: ${{ workflow.workflow_details.return }}
    widgets:
      - ${{ workflow.map_widget_merged.return }}
    groupers: ${{ workflow.groupers.return }}
    time_range: ${{ workflow.time_range.return }}
```

---

## Next steps

- **[Form Customization](./form-customization.md)** — Control which parameters are exposed and how they appear.
- **[indexes reference](../reference/indexes.md)** — Full API for temporal indexes, `AllGrouper`, `ValueGrouper`, and other grouper types.
- **[wt Tutorials](https://wt.readthedocs.io/en/latest/tutorials/)** — Deep dive into `map`, `mapvalues`, and `groupbykey`.
