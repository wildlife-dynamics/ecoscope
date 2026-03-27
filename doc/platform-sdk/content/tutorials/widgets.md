# Widgets

Widgets are the visual building blocks of a workflow dashboard. This tutorial shows how to create each widget type, assemble them into a dashboard, and control their layout.

Cross-references: [results tasks reference](../reference/tasks/results.md)

---

## The widget pattern

Every widget follows the same pipeline:

```
compute data  →  render to HTML  →  persist HTML  →  create widget  →  merge views
```

1. A task computes the data (e.g., `draw_ecomap` renders a map).
2. `persist_text` saves the HTML string to a file and returns a URL.
3. A `create_*_widget_single_view` task wraps the URL into a widget object.
4. `merge_widget_views` combines grouped views (for ungrouped workflows, it simply passes through).

Widget creation tasks have a `view` parameter. For ungrouped workflows, use `view: null` or `view: ["All", "=", "True"]`. Groupers are covered in the [Groupers tutorial](./groupers.md).

---

## Map widget

A map widget displays an interactive EcoMap with one or more layers.

```yaml
- id: point_layer
  task: create_point_layer
  partial:
    geodataframe: ${{ workflow.colored_data.return }}
    layer_style:
      fill_color_column: color
      get_radius: 5

- id: ecomap
  task: draw_ecomap
  partial:
    geo_layers: ${{ workflow.point_layer.return }}
    tile_layers:
      - layer_name: "TERRAIN"

- id: ecomap_html
  task: persist_text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
    text: ${{ workflow.ecomap.return }}

- id: map_widget
  task: create_map_widget_single_view
  skipif:
    conditions:
      - never
  partial:
    title: "My Map"
    view: null
    data: ${{ workflow.ecomap_html.return }}
```

**Layer types**: `create_point_layer`, `create_polyline_layer`, `create_polygon_layer`, `create_text_layer`. You can pass multiple layers to `draw_ecomap` as a list.

---

## Plot widget

Plot widgets display Plotly charts. Available chart tasks:

| Task | Chart type |
|------|-----------|
| `draw_time_series_bar_chart` | Stacked time-series bars |
| `draw_bar_chart` | Grouped bar chart |
| `draw_pie_chart` | Pie chart |
| `draw_line_chart` | Line chart |
| `draw_ecoplot` | Multi-trace EcoPlot |
| `draw_historic_timeseries` | Current vs. historic band |

```yaml
- id: pie_chart
  task: draw_pie_chart
  partial:
    dataframe: ${{ workflow.summary_data.return }}
    value_column: count
    label_column: category

- id: pie_html
  task: persist_text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
    text: ${{ workflow.pie_chart.return }}

- id: pie_widget
  task: create_plot_widget_single_view
  skipif:
    conditions:
      - never
  partial:
    title: "Category Distribution"
    view: null
    data: ${{ workflow.pie_html.return }}
```

---

## Table widget

```yaml
- id: table
  task: draw_table
  partial:
    dataframe: ${{ workflow.summary_data.return }}
    columns:
      - name
      - count
      - percentage

- id: table_html
  task: persist_text
  partial:
    root_path: ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
    text: ${{ workflow.table.return }}

- id: table_widget
  task: create_table_widget_single_view
  skipif:
    conditions:
      - never
  partial:
    title: "Summary Table"
    view: null
    data: ${{ workflow.table_html.return }}
```

---

## Single-value widget

Single-value widgets display a number — a count, mean, total, etc. They skip the render/persist steps because the data is just a scalar:

```yaml
- id: event_count
  task: dataframe_count
  partial:
    df: ${{ workflow.get_events_data.return }}

- id: count_widget
  task: create_single_value_widget_single_view
  skipif:
    conditions:
      - never
  partial:
    title: "Total Events"
    view: null
    data: ${{ workflow.event_count.return }}
```

---

## Text widget

Text widgets display a string. Like single-value widgets, no render/persist step is needed:

```yaml
- id: text_widget
  task: create_text_widget_single_view
  skipif:
    conditions:
      - never
  partial:
    title: "Notes"
    view: null
    data: "This workflow analyzes event occurrences."
```

---

## Dashboard assembly

`gather_dashboard` collects all widgets into the final dashboard:

```yaml
- id: dashboard
  task: gather_dashboard
  partial:
    details: ${{ workflow.workflow_details.return }}
    widgets:
      - ${{ workflow.map_widget.return }}
      - ${{ workflow.pie_widget.return }}
      - ${{ workflow.table_widget.return }}
      - ${{ workflow.count_widget.return }}
    groupers:
      - index_name: "All"
    time_range: ${{ workflow.time_range.return }}
```

The order of `widgets` matters — it determines the `widget_id` used in `layout.json` (0-indexed).

---

## `layout.json` — The dashboard grid

`layout.json` controls how widgets are arranged in the dashboard. The grid has **10 columns** and grows vertically as needed.

Each entry maps a widget to a grid position:

```json
[
  {
    "i": 0,
    "widget_id": 0,
    "x": 0,
    "y": 0,
    "w": 5,
    "h": 8,
    "minW": 3,
    "static": false
  },
  {
    "i": 1,
    "widget_id": 1,
    "x": 5,
    "y": 0,
    "w": 5,
    "h": 8,
    "minW": 3,
    "static": false
  },
  {
    "i": 2,
    "widget_id": 2,
    "x": 0,
    "y": 8,
    "w": 7,
    "h": 6,
    "minW": 4,
    "static": false
  },
  {
    "i": 3,
    "widget_id": 3,
    "x": 7,
    "y": 8,
    "w": 3,
    "h": 6,
    "minW": 2,
    "static": false
  }
]
```

| Field | Description |
|-------|-------------|
| `i` | Layout slot index (must be unique, typically matches array position) |
| `widget_id` | Index into the `gather_dashboard.widgets` array (0-based) |
| `x` | Column position (0-9) |
| `y` | Row position (grows downward) |
| `w` | Width in columns |
| `h` | Height in rows |
| `minW` | Minimum width (prevents users from shrinking below this) |
| `static` | If `true`, the widget cannot be moved or resized |

**Mapping widgets to the grid**: If `gather_dashboard.widgets` lists `[map, pie, table, count]`, then `widget_id: 0` is the map, `widget_id: 1` is the pie chart, and so on.

The events-map-example uses a single full-width widget:

```json
[{ "i": 0, "widget_id": 0, "x": 0, "y": 0, "w": 10, "h": 10, "minW": 5, "static": false }]
```

---

## Next steps

- **[Groupers](./groupers.md)** — Create multi-view dashboards by slicing data into groups.
- **[results tasks reference](../reference/tasks/results.md)** — Full API for all visualization and widget tasks.
