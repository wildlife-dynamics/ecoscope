# Built-in Tasks

The Platform SDK ships ~80 tasks ready to use in your `spec.yaml`. This page helps you find the right one by use case. Each entry links to the full API reference.

---

## Loading data

| I need to... | Task | Reference |
|--------------|------|-----------|
| Load events from EarthRanger | `get_events` | [io](reference/tasks/io.md) |
| Load events with combined params | `get_events_from_combined_params` | [io](reference/tasks/io.md) |
| Load patrol events | `get_patrol_events` | [io](reference/tasks/io.md) |
| Load patrols | `get_patrols` | [io](reference/tasks/io.md) |
| Load patrol observations | `get_patrol_observations` | [io](reference/tasks/io.md) |
| Load subject group GPS data | `get_subjectgroup_observations` | [io](reference/tasks/io.md) |
| Load events from SMART | `get_events_from_smart` | [io](reference/tasks/io.md) |
| Load patrol observations from SMART | `get_patrol_observations_from_smart` | [io](reference/tasks/io.md) |
| Download a region from Earth Engine | `download_roi` | [io](reference/tasks/io.md) |
| Calculate NDVI range | `calculate_ndvi_range` | [io](reference/tasks/io.md) |
| Get event type display names | `get_event_type_display_names_from_events` | [io](reference/tasks/io.md) |
| Get spatial feature groups | `get_spatial_features_group` | [io](reference/tasks/io.md) |

## Connections

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set up an EarthRanger connection | `set_er_connection` | [io](reference/tasks/io.md) |
| Set up a SMART connection | `set_smart_connection` | [io](reference/tasks/io.md) |
| Set up an Earth Engine connection | `set_gee_connection` | [io](reference/tasks/io.md) |

## Filtering

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set a time range | `set_time_range` | [filter](reference/tasks/filter.md) |
| Get timezone from a time range | `get_timezone_from_time_range` | [filter](reference/tasks/filter.md) |
| Filter rows by a condition | `filter_df` | [transformation](reference/tasks/transformation.md) |
| Drop rows with NaN in a column | `drop_nan_values_by_column` | [transformation](reference/tasks/transformation.md) |
| Drop rows with null geometry | `drop_null_geometry` | [transformation](reference/tasks/transformation.md) |
| Apply a relocation coordinate filter | `apply_reloc_coord_filter` | [transformation](reference/tasks/transformation.md) |

## Preprocessing

| I need to... | Task | Reference |
|--------------|------|-----------|
| Process relocations from GPS fixes | `process_relocations` | [preprocessing](reference/tasks/preprocessing.md) |
| Convert relocations to trajectories | `relocations_to_trajectory` | [preprocessing](reference/tasks/preprocessing.md) |
| Filter trajectory segments | `TrajectorySegmentFilter` | [preprocessing](reference/tasks/preprocessing.md) |

## Transformation

| I need to... | Task | Reference |
|--------------|------|-----------|
| Apply a colormap to a column | `apply_color_map` | [transformation](reference/tasks/transformation.md) |
| Classify day vs. night | `classify_is_night` | [transformation](reference/tasks/transformation.md) |
| Classify by season | `classify_seasons` | [transformation](reference/tasks/transformation.md) |
| Apply a custom classification | `apply_classification` | [transformation](reference/tasks/transformation.md) |
| Add a temporal index column | `add_temporal_index` | [transformation](reference/tasks/transformation.md) |
| Add a spatial index column | `add_spatial_index` | [transformation](reference/tasks/transformation.md) |
| Convert column values to numeric | `convert_column_values_to_numeric` | [transformation](reference/tasks/transformation.md) |
| Convert column values to string | `convert_column_values_to_string` | [transformation](reference/tasks/transformation.md) |
| Convert values to a timezone | `convert_values_to_timezone` | [transformation](reference/tasks/transformation.md) |
| Extract a column as a specific type | `extract_column_as_type` | [transformation](reference/tasks/transformation.md) |
| Extract values from a JSON column | `extract_value_from_json_column` | [transformation](reference/tasks/transformation.md) |
| Normalize a JSON column | `normalize_json_column` | [transformation](reference/tasks/transformation.md) |
| Normalize a numeric column | `normalize_numeric_column` | [transformation](reference/tasks/transformation.md) |
| Rename columns | `map_columns` (via `RenameColumn`) | [transformation](reference/tasks/transformation.md) |
| Map values in a column | `map_values` | [transformation](reference/tasks/transformation.md) |
| Sort rows | `sort_values` | [transformation](reference/tasks/transformation.md) |
| Transpose a DataFrame | `transpose` | [transformation](reference/tasks/transformation.md) |
| Explode a list column | `explode` | [transformation](reference/tasks/transformation.md) |
| Fill NaN values | `fill_na` | [transformation](reference/tasks/transformation.md) |
| Assign a fixed value to a column | `assign_value` | [transformation](reference/tasks/transformation.md) |
| Assign subject colors | `assign_subject_colors` | [transformation](reference/tasks/transformation.md) |

## Grouping

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set grouper definitions | `set_groupers` | [groupby](reference/tasks/groupby.md) |
| Split data into groups | `split_groups` | [groupby](reference/tasks/groupby.md) |
| Combine keyed iterables | `groupbykey` | [groupby](reference/tasks/groupby.md) |
| Merge grouped DataFrames | `merge_df` | [groupby](reference/tasks/groupby.md) |

## Analysis

| I need to... | Task | Reference |
|--------------|------|-----------|
| Count rows | `dataframe_count` | [analysis](reference/tasks/analysis.md) |
| Sum a column | `dataframe_column_sum` | [analysis](reference/tasks/analysis.md) |
| Average a column | `dataframe_column_mean` | [analysis](reference/tasks/analysis.md) |
| Find the max of a column | `dataframe_column_max` | [analysis](reference/tasks/analysis.md) |
| Find the min of a column | `dataframe_column_min` | [analysis](reference/tasks/analysis.md) |
| Count unique values | `dataframe_column_nunique` | [analysis](reference/tasks/analysis.md) |
| Get the first unique value | `dataframe_column_first_unique` | [analysis](reference/tasks/analysis.md) |
| Calculate a percentile | `dataframe_column_percentile` | [analysis](reference/tasks/analysis.md) |
| Apply arithmetic operations | `apply_arithmetic_operation` | [analysis](reference/tasks/analysis.md) |
| Summarize a DataFrame | `summarize_df` | [analysis](reference/tasks/analysis.md) |
| Aggregate over rows | `aggregate_over_rows` | [analysis](reference/tasks/analysis.md) |
| Calculate night/day ratio | `get_night_day_ratio` | [analysis](reference/tasks/analysis.md) |
| Calculate feature density | `calculate_feature_density` | [analysis](reference/tasks/analysis.md) |
| Calculate elliptical time density | `calculate_elliptical_time_density` | [analysis](reference/tasks/analysis.md) |
| Calculate linear time density | `calculate_linear_time_density` | [analysis](reference/tasks/analysis.md) |
| Create a meshgrid | `create_meshgrid` | [analysis](reference/tasks/analysis.md) |

## Visualization

| I need to... | Task | Reference |
|--------------|------|-----------|
| Draw an interactive map | `draw_ecomap` | [results](reference/tasks/results.md) |
| Create a point layer | `create_point_layer` | [results](reference/tasks/results.md) |
| Create a polyline layer | `create_polyline_layer` | [results](reference/tasks/results.md) |
| Create a polygon layer | `create_polygon_layer` | [results](reference/tasks/results.md) |
| Create a text layer | `create_text_layer` | [results](reference/tasks/results.md) |
| Set base maps | `set_base_maps` | [results](reference/tasks/results.md) |
| Draw a time-series bar chart | `draw_time_series_bar_chart` | [results](reference/tasks/results.md) |
| Draw a bar chart | `draw_bar_chart` | [results](reference/tasks/results.md) |
| Draw a pie chart | `draw_pie_chart` | [results](reference/tasks/results.md) |
| Draw a line chart | `draw_line_chart` | [results](reference/tasks/results.md) |
| Draw an EcoPlot | `draw_ecoplot` | [results](reference/tasks/results.md) |
| Draw a historic timeseries | `draw_historic_timeseries` | [results](reference/tasks/results.md) |
| Draw an HTML table | `draw_table` | [results](reference/tasks/results.md) |

## Dashboard and Widgets

| I need to... | Task | Reference |
|--------------|------|-----------|
| Create a map widget | `create_map_widget_single_view` | [results](reference/tasks/results.md) |
| Create a plot widget | `create_plot_widget_single_view` | [results](reference/tasks/results.md) |
| Create a table widget | `create_table_widget_single_view` | [results](reference/tasks/results.md) |
| Create a single-value widget | `create_single_value_widget_single_view` | [results](reference/tasks/results.md) |
| Create a text widget | `create_text_widget_single_view` | [results](reference/tasks/results.md) |
| Merge grouped widget views | `merge_widget_views` | [results](reference/tasks/results.md) |
| Assemble the final dashboard | `gather_dashboard` | [results](reference/tasks/results.md) |
| Gather output files | `gather_output_files` | [results](reference/tasks/results.md) |

## Persistence

| I need to... | Task | Reference |
|--------------|------|-----------|
| Persist an HTML string | `persist_text` | [io](reference/tasks/io.md) |
| Persist a DataFrame | `persist_df` | [io](reference/tasks/io.md) |

## Workflow Utilities

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set workflow name and description | `set_workflow_details` | [config](reference/tasks/config.md) |
| Set a string variable | `set_string_var` | [config](reference/tasks/config.md) |
| Set a boolean variable | `set_bool_var` | [config](reference/tasks/config.md) |
| Concatenate strings | `concat_string_vars` | [config](reference/tasks/config.md) |
| Title-case a string | `title_case_var` | [config](reference/tasks/config.md) |
| Get column names from a DataFrame | `get_column_names_from_dataframe` | [config](reference/tasks/config.md) |

## Skip Conditions

| I need to... | Condition | Reference |
|--------------|-----------|-----------|
| Skip if any DataFrame input is empty | `any_is_empty_df` | [skip](reference/tasks/skip.md) |
| Skip if any dependency was skipped | `any_dependency_skipped` | [skip](reference/tasks/skip.md) |
| Skip if all geometries are null | `all_geometry_are_none` | [skip](reference/tasks/skip.md) |
| Skip if any dependency is None | `any_dependency_is_none` | [skip](reference/tasks/skip.md) |
| Never skip (override defaults) | `never` | [skip](reference/tasks/skip.md) |
| Fall back GeoDataFrame to None on skip | `skip_gdf_fallback_to_none` | [skip](reference/tasks/skip.md) |
