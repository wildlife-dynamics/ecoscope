"""Skip-condition predicates for conditional task execution.

Each function in this module evaluates whether a task should be skipped based
on its inputs (e.g., empty DataFrames, None dependencies, missing geometry).
Use these as ``skip_if`` callbacks in workflow task definitions.
"""

from ._skip import (
    all_geometry_are_none,
    all_keyed_iterables_are_skips,
    any_dependency_is_empty_string,
    any_dependency_is_none,
    any_dependency_skipped,
    any_is_empty_df,
    never,
    skip_gdf_fallback_to_none,
)

__all__ = [
    "all_geometry_are_none",
    "all_keyed_iterables_are_skips",
    "any_dependency_is_empty_string",
    "any_dependency_is_none",
    "any_dependency_skipped",
    "any_is_empty_df",
    "never",
    "skip_gdf_fallback_to_none",
]
