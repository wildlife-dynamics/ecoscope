# Groupers

Groupers let you slice data into multiple views — for example, by month, by species, or by patrol area. The dashboard shows a dropdown so users can switch between views.

Cross-references: [indexes reference](../reference/indexes.md), [groupby tasks reference](../reference/tasks/groupby.md)

---

## What is a grouper?

A grouper splits a DataFrame into subsets by some criterion. Each subset becomes a separate **view** in the dashboard. The general pattern is:

```
set_groupers  →  split_groups  →  (process each group with mapvalues)  →  merge views
```

[`set_groupers`][ecoscope.platform.tasks.groupby.set_groupers] accepts user input for how to group the data. [`split_groups`][ecoscope.platform.tasks.groupby.split_groups] takes a DataFrame and the grouper definition, and returns a **keyed iterable** — a list of 2-tuples where each key is a composite filter (identifying the group) and each value is the subset of data.

---

## `map` vs `mapvalues` — Two kinds of iteration

The `wt-compiler` provides two operators for running a task over a collection. See [`map`](https://wt.readthedocs.io/en/latest/reference/spec-yaml/#map) and [`mapvalues`](https://wt.readthedocs.io/en/latest/reference/spec-yaml/#mapvalues) in the wt reference. Understanding the difference is essential for grouped workflows.

**`map`** iterates over any sequence, applying the task function to each element and producing a list of results. The `argnames` field lists which parameter(s) to iterate over, and `argvalues` provides the iterable source:

```yaml
- id: persist_all
  task: persist_text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
  map:
    argnames: text
    argvalues: ${{ workflow.all_htmls.return }}
```

If `all_htmls` returns `[html_a, html_b, html_c]`, then `persist_text` runs three times, once per element, and the result is a list of three URLs.

**`mapvalues`** also iterates over a sequence, but assumes each element is a 2-tuple `(key, value)`. It applies the function to the **value** while preserving the **key**:

```yaml
- id: color_per_group
  task: apply_color_map
  partial:
    input_column_name: event_type
    colormap: tab20b
  mapvalues:
    argnames: df
    argvalues: ${{ workflow.split_data.return }}
```

If `split_data` returns `[(filter_a, df_a), (filter_b, df_b)]`, then `apply_color_map` runs twice — once on `df_a` and once on `df_b` — and the result is `[(filter_a, mapped_a), (filter_b, mapped_b)]`. The filter keys are preserved.

**The key insight**: after `split_groups`, data is a keyed iterable, so you use **`mapvalues`** for all per-group processing. At the widget creation step, the pipeline transitions to **`map`** because `create_*_widget_single_view` takes `[view, data]` — the view key and the data value as separate arguments.

For a deep dive into these operators, see the [wt Tutorials](https://wt.readthedocs.io/en/latest/tutorials/).

---

## AllGrouper (default)

[`AllGrouper`][ecoscope.platform.indexes.AllGrouper] is the simplest grouper — it creates a single view containing all data.

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

With `AllGrouper`, `split_groups` returns a single-entry keyed iterable: `[("All", "=", "True"), <full dataframe>]`. The rest of the pipeline processes this one group normally.

---

## ValueGrouper

[`ValueGrouper`][ecoscope.platform.indexes.ValueGrouper] groups data by the distinct values in a categorical column. In the configuration form, this is displayed as "Category" — any category the user inputs must correspond to a categorical (i.e., string) column in the DataFrame to be processed.

```yaml
- id: groupers
  task: set_groupers
```

By default, the configuration form exposes all grouper options to the user. If the user selects `event_type` as the category, `split_groups` creates one view per unique event type value. The dashboard dropdown lists each event type.

---

## TemporalGrouper

[`TemporalGrouper`][ecoscope.platform.indexes.TemporalGrouper] groups data by time periods. The Platform SDK provides several temporal index types:

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
```

By default, the configuration form exposes all grouper options to the user, including temporal grouping. If the user selects a temporal grouper with the `Month` index, `split_groups` creates one view per month present in the data.

---

## SpatialGrouper

[`SpatialGrouper`][ecoscope.platform.indexes.SpatialGrouper] groups data by geographic regions (e.g., conservation areas, patrol zones). It requires a spatial feature group from the data source to define the regions. This grouper is currently limited to spatial feature groups defined in EarthRanger.

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

# 2. Create layers per group (mapvalues — keyed iterable in, keyed iterable out)
- id: point_layers
  task: create_point_layer
  partial:
    layer_style:
      fill_color_column: color
      get_radius: 5
  mapvalues:
    argnames: geodataframe
    argvalues: ${{ workflow.split_data.return }}

# 3. Draw maps per group
- id: ecomaps
  task: draw_ecomap
  partial:
    tile_layers:
      - layer_name: "TERRAIN"
  mapvalues:
    argnames: geo_layers
    argvalues: ${{ workflow.point_layers.return }}

# 4. Persist HTML per group
- id: ecomap_htmls
  task: persist_text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
  mapvalues:
    argnames: text
    argvalues: ${{ workflow.ecomaps.return }}

# 5. Create widgets (map — transition from keyed iterable to list)
- id: map_widgets
  task: create_map_widget_single_view
  skipif:
    conditions:
      - never
  partial:
    title: "Events Map"
  map:
    argnames:
      - view
      - data
    argvalues: ${{ workflow.ecomap_htmls.return }}

# 6. Merge views into a single grouped widget
- id: map_widget_merged
  task: merge_widget_views
  partial:
    widgets: ${{ workflow.map_widgets.return }}
```

The key transition happens at step 5: `create_map_widget_single_view` uses `map` with `argnames: [view, data]` to unpack the keyed iterable into separate widget views. Then `merge_widget_views` combines them into a single grouped widget with a view dropdown.

See `GroupedWidget`, `WidgetSingleView`, and [`merge_widget_views`][ecoscope.platform.tasks.results.merge_widget_views] in the Reference.

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
