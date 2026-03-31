# Built-in Tasks

The Platform SDK ships ~80 tasks ready to use in your `spec.yaml`. This page helps you find the right one by use case. Each entry links to the full API reference.

---

## Loading data

| I need to... | Task | Reference |
|--------------|------|-----------|
| Load events from EarthRanger | `get_events` | [`get_events`][ecoscope.platform.tasks.io.get_events] |
| Load events with combined params | `get_events_from_combined_params` | [`get_events_from_combined_params`][ecoscope.platform.tasks.io.get_events_from_combined_params] |
| Load patrol events | `get_patrol_events` | [`get_patrol_events`][ecoscope.platform.tasks.io.get_patrol_events] |
| Load patrols | `get_patrols` | [`get_patrols`][ecoscope.platform.tasks.io.get_patrols] |
| Load patrol observations | `get_patrol_observations` | [`get_patrol_observations`][ecoscope.platform.tasks.io.get_patrol_observations] |
| Load subject group GPS data | `get_subjectgroup_observations` | [`get_subjectgroup_observations`][ecoscope.platform.tasks.io.get_subjectgroup_observations] |
| Load events from SMART | `get_events_from_smart` | [`get_events_from_smart`][ecoscope.platform.tasks.io.get_events_from_smart] |
| Load patrol observations from SMART | `get_patrol_observations_from_smart` | [`get_patrol_observations_from_smart`][ecoscope.platform.tasks.io.get_patrol_observations_from_smart] |
| Download a region from Earth Engine | `download_roi` | [`download_roi`][ecoscope.platform.tasks.io.download_roi] |
| Calculate NDVI range | `calculate_ndvi_range` | [`calculate_ndvi_range`][ecoscope.platform.tasks.io.calculate_ndvi_range] |
| Get event type display names | `get_event_type_display_names_from_events` | [`get_event_type_display_names_from_events`][ecoscope.platform.tasks.io.get_event_type_display_names_from_events] |
| Get spatial feature groups | `get_spatial_features_group` | [`get_spatial_features_group`][ecoscope.platform.tasks.io.get_spatial_features_group] |

## Connections

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set up an EarthRanger connection | `set_er_connection` | [`set_er_connection`][ecoscope.platform.tasks.io.set_er_connection] |
| Set up a SMART connection | `set_smart_connection` | [`set_smart_connection`][ecoscope.platform.tasks.io.set_smart_connection] |
| Set up an Earth Engine connection | `set_gee_connection` | [`set_gee_connection`][ecoscope.platform.tasks.io.set_gee_connection] |

## Filtering

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set a time range | `set_time_range` | [`set_time_range`][ecoscope.platform.tasks.filter.set_time_range] |
| Get timezone from a time range | `get_timezone_from_time_range` | [`get_timezone_from_time_range`][ecoscope.platform.tasks.filter.get_timezone_from_time_range] |
| Filter rows by a condition | `filter_df` | [`filter_df`][ecoscope.platform.tasks.transformation.filter_df] |
| Drop rows with NaN in a column | `drop_nan_values_by_column` | [`drop_nan_values_by_column`][ecoscope.platform.tasks.transformation.drop_nan_values_by_column] |
| Drop rows with null geometry | `drop_null_geometry` | [`drop_null_geometry`][ecoscope.platform.tasks.transformation.drop_null_geometry] |
| Apply a relocation coordinate filter | `apply_reloc_coord_filter` | [`apply_reloc_coord_filter`][ecoscope.platform.tasks.transformation.apply_reloc_coord_filter] |

## Preprocessing

| I need to... | Task | Reference |
|--------------|------|-----------|
| Process relocations from GPS fixes | `process_relocations` | [`process_relocations`][ecoscope.platform.tasks.preprocessing.process_relocations] |
| Convert relocations to trajectories | `relocations_to_trajectory` | [`relocations_to_trajectory`][ecoscope.platform.tasks.preprocessing.relocations_to_trajectory] |
| Filter trajectory segments | `TrajectorySegmentFilter` | [`TrajectorySegmentFilter`][ecoscope.platform.tasks.preprocessing.TrajectorySegmentFilter] |

## Transformation

| I need to... | Task | Reference |
|--------------|------|-----------|
| Apply a colormap to a column | `apply_color_map` | [`apply_color_map`][ecoscope.platform.tasks.transformation.apply_color_map] |
| Classify day vs. night | `classify_is_night` | [`classify_is_night`][ecoscope.platform.tasks.transformation.classify_is_night] |
| Classify by season | `classify_seasons` | [`classify_seasons`][ecoscope.platform.tasks.transformation.classify_seasons] |
| Apply a custom classification | `apply_classification` | [`apply_classification`][ecoscope.platform.tasks.transformation.apply_classification] |
| Add a temporal index column | `add_temporal_index` | [`add_temporal_index`][ecoscope.platform.tasks.transformation.add_temporal_index] |
| Add a spatial index column | `add_spatial_index` | [`add_spatial_index`][ecoscope.platform.tasks.transformation.add_spatial_index] |
| Convert column values to numeric | `convert_column_values_to_numeric` | [`convert_column_values_to_numeric`][ecoscope.platform.tasks.transformation.convert_column_values_to_numeric] |
| Convert column values to string | `convert_column_values_to_string` | [`convert_column_values_to_string`][ecoscope.platform.tasks.transformation.convert_column_values_to_string] |
| Convert values to a timezone | `convert_values_to_timezone` | [`convert_values_to_timezone`][ecoscope.platform.tasks.transformation.convert_values_to_timezone] |
| Extract a column as a specific type | `extract_column_as_type` | [`extract_column_as_type`][ecoscope.platform.tasks.transformation.extract_column_as_type] |
| Extract values from a JSON column | `extract_value_from_json_column` | [`extract_value_from_json_column`][ecoscope.platform.tasks.transformation.extract_value_from_json_column] |
| Normalize a JSON column | `normalize_json_column` | [`normalize_json_column`][ecoscope.platform.tasks.transformation.normalize_json_column] |
| Normalize a numeric column | `normalize_numeric_column` | [`normalize_numeric_column`][ecoscope.platform.tasks.transformation.normalize_numeric_column] |
| Rename columns | `map_columns` (via `RenameColumn`) | [`map_columns`][ecoscope.platform.tasks.transformation.map_columns] |
| Map values in a column | `map_values` | [`map_values`][ecoscope.platform.tasks.transformation.map_values] |
| Sort rows | `sort_values` | [`sort_values`][ecoscope.platform.tasks.transformation.sort_values] |
| Transpose a DataFrame | `transpose` | [`transpose`][ecoscope.platform.tasks.transformation.transpose] |
| Explode a list column | `explode` | [`explode`][ecoscope.platform.tasks.transformation.explode] |
| Fill NaN values | `fill_na` | [`fill_na`][ecoscope.platform.tasks.transformation.fill_na] |
| Assign a fixed value to a column | `assign_value` | [`assign_value`][ecoscope.platform.tasks.transformation.assign_value] |
| Assign subject colors | `assign_subject_colors` | [`assign_subject_colors`][ecoscope.platform.tasks.transformation.assign_subject_colors] |

## Grouping

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set grouper definitions | `set_groupers` | [`set_groupers`][ecoscope.platform.tasks.groupby.set_groupers] |
| Split data into groups | `split_groups` | [`split_groups`][ecoscope.platform.tasks.groupby.split_groups] |
| Combine keyed iterables | `groupbykey` | [`groupbykey`][ecoscope.platform.tasks.groupby.groupbykey] |
| Merge grouped DataFrames | `merge_df` | [`merge_df`][ecoscope.platform.tasks.groupby.merge_df] |

## Analysis

| I need to... | Task | Reference |
|--------------|------|-----------|
| Count rows | `dataframe_count` | [`dataframe_count`][ecoscope.platform.tasks.analysis.dataframe_count] |
| Sum a column | `dataframe_column_sum` | [`dataframe_column_sum`][ecoscope.platform.tasks.analysis.dataframe_column_sum] |
| Average a column | `dataframe_column_mean` | [`dataframe_column_mean`][ecoscope.platform.tasks.analysis.dataframe_column_mean] |
| Find the max of a column | `dataframe_column_max` | [`dataframe_column_max`][ecoscope.platform.tasks.analysis.dataframe_column_max] |
| Find the min of a column | `dataframe_column_min` | [`dataframe_column_min`][ecoscope.platform.tasks.analysis.dataframe_column_min] |
| Count unique values | `dataframe_column_nunique` | [`dataframe_column_nunique`][ecoscope.platform.tasks.analysis.dataframe_column_nunique] |
| Get the first unique value | `dataframe_column_first_unique` | [`dataframe_column_first_unique`][ecoscope.platform.tasks.analysis.dataframe_column_first_unique] |
| Calculate a percentile | `dataframe_column_percentile` | [`dataframe_column_percentile`][ecoscope.platform.tasks.analysis.dataframe_column_percentile] |
| Apply arithmetic operations | `apply_arithmetic_operation` | [`apply_arithmetic_operation`][ecoscope.platform.tasks.analysis.apply_arithmetic_operation] |
| Summarize a DataFrame | `summarize_df` | [`summarize_df`][ecoscope.platform.tasks.analysis.summarize_df] |
| Aggregate over rows | `aggregate_over_rows` | [`aggregate_over_rows`][ecoscope.platform.tasks.analysis.aggregate_over_rows] |
| Calculate night/day ratio | `get_night_day_ratio` | [`get_night_day_ratio`][ecoscope.platform.tasks.analysis.get_night_day_ratio] |
| Calculate feature density | `calculate_feature_density` | [`calculate_feature_density`][ecoscope.platform.tasks.analysis.calculate_feature_density] |
| Calculate elliptical time density | `calculate_elliptical_time_density` | [`calculate_elliptical_time_density`][ecoscope.platform.tasks.analysis.calculate_elliptical_time_density] |
| Calculate linear time density | `calculate_linear_time_density` | [`calculate_linear_time_density`][ecoscope.platform.tasks.analysis.calculate_linear_time_density] |
| Create a meshgrid | `create_meshgrid` | [`create_meshgrid`][ecoscope.platform.tasks.analysis.create_meshgrid] |

## Visualization

| I need to... | Task | Reference |
|--------------|------|-----------|
| Draw an interactive map | `draw_ecomap` | [`draw_ecomap`][ecoscope.platform.tasks.results.draw_ecomap] |
| Create a point layer | `create_point_layer` | [`create_point_layer`][ecoscope.platform.tasks.results.create_point_layer] |
| Create a polyline layer | `create_polyline_layer` | [`create_polyline_layer`][ecoscope.platform.tasks.results.create_polyline_layer] |
| Create a polygon layer | `create_polygon_layer` | [`create_polygon_layer`][ecoscope.platform.tasks.results.create_polygon_layer] |
| Create a text layer | `create_text_layer` | [`create_text_layer`][ecoscope.platform.tasks.results.create_text_layer] |
| Set base maps | `set_base_maps` | [`set_base_maps`][ecoscope.platform.tasks.results.set_base_maps] |
| Draw a time-series bar chart | `draw_time_series_bar_chart` | [`draw_time_series_bar_chart`][ecoscope.platform.tasks.results.draw_time_series_bar_chart] |
| Draw a bar chart | `draw_bar_chart` | [`draw_bar_chart`][ecoscope.platform.tasks.results.draw_bar_chart] |
| Draw a pie chart | `draw_pie_chart` | [`draw_pie_chart`][ecoscope.platform.tasks.results.draw_pie_chart] |
| Draw a line chart | `draw_line_chart` | [`draw_line_chart`][ecoscope.platform.tasks.results.draw_line_chart] |
| Draw an EcoPlot | `draw_ecoplot` | [`draw_ecoplot`][ecoscope.platform.tasks.results.draw_ecoplot] |
| Draw a historic timeseries | `draw_historic_timeseries` | [`draw_historic_timeseries`][ecoscope.platform.tasks.results.draw_historic_timeseries] |
| Draw an HTML table | `draw_table` | [`draw_table`][ecoscope.platform.tasks.results.draw_table] |

## Dashboard and Widgets

| I need to... | Task | Reference |
|--------------|------|-----------|
| Create a map widget | `create_map_widget_single_view` | [`create_map_widget_single_view`][ecoscope.platform.tasks.results.create_map_widget_single_view] |
| Create a plot widget | `create_plot_widget_single_view` | [`create_plot_widget_single_view`][ecoscope.platform.tasks.results.create_plot_widget_single_view] |
| Create a table widget | `create_table_widget_single_view` | [`create_table_widget_single_view`][ecoscope.platform.tasks.results.create_table_widget_single_view] |
| Create a single-value widget | `create_single_value_widget_single_view` | [`create_single_value_widget_single_view`][ecoscope.platform.tasks.results.create_single_value_widget_single_view] |
| Create a text widget | `create_text_widget_single_view` | [`create_text_widget_single_view`][ecoscope.platform.tasks.results.create_text_widget_single_view] |
| Merge grouped widget views | `merge_widget_views` | [`merge_widget_views`][ecoscope.platform.tasks.results.merge_widget_views] |
| Assemble the final dashboard | `gather_dashboard` | [`gather_dashboard`][ecoscope.platform.tasks.results.gather_dashboard] |
| Gather output files | `gather_output_files` | [`gather_output_files`][ecoscope.platform.tasks.results.gather_output_files] |

## Persistence

| I need to... | Task | Reference |
|--------------|------|-----------|
| Persist an HTML string | `persist_text` | [`persist_text`][ecoscope.platform.tasks.io.persist_text] |
| Persist a DataFrame | `persist_df` | [`persist_df`][ecoscope.platform.tasks.io.persist_df] |

## Workflow Utilities

| I need to... | Task | Reference |
|--------------|------|-----------|
| Set workflow name and description | `set_workflow_details` | [`set_workflow_details`][ecoscope.platform.tasks.config.set_workflow_details] |
| Set a string variable | `set_string_var` | [`set_string_var`][ecoscope.platform.tasks.config.set_string_var] |
| Set a boolean variable | `set_bool_var` | [`set_bool_var`][ecoscope.platform.tasks.config.set_bool_var] |
| Concatenate strings | `concat_string_vars` | [`concat_string_vars`][ecoscope.platform.tasks.config.concat_string_vars] |
| Title-case a string | `title_case_var` | [`title_case_var`][ecoscope.platform.tasks.config.title_case_var] |
| Get column names from a DataFrame | `get_column_names_from_dataframe` | [`get_column_names_from_dataframe`][ecoscope.platform.tasks.config.get_column_names_from_dataframe] |

## Skip Conditions

| I need to... | Condition | Reference |
|--------------|-----------|-----------|
| Skip if any DataFrame input is empty | `any_is_empty_df` | [`any_is_empty_df`][ecoscope.platform.tasks.skip.any_is_empty_df] |
| Skip if any dependency was skipped | `any_dependency_skipped` | [`any_dependency_skipped`][ecoscope.platform.tasks.skip.any_dependency_skipped] |
| Skip if all geometries are null | `all_geometry_are_none` | [`all_geometry_are_none`][ecoscope.platform.tasks.skip.all_geometry_are_none] |
| Skip if any dependency is None | `any_dependency_is_none` | [`any_dependency_is_none`][ecoscope.platform.tasks.skip.any_dependency_is_none] |
| Never skip (override defaults) | `never` | [`never`][ecoscope.platform.tasks.skip.never] |
| Fall back GeoDataFrame to None on skip | `skip_gdf_fallback_to_none` | [`skip_gdf_fallback_to_none`][ecoscope.platform.tasks.skip.skip_gdf_fallback_to_none] |
