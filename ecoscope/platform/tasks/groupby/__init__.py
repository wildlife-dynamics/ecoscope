"""Group-by operations for splitting, keying, and merging DataFrames.

Use this module to configure user-defined groupers, split a DataFrame into
keyed groups, merge grouped results back together, and generate group keys.
"""

from ._groupby import (
    UserDefinedGroupers,
    groupbykey,
    merge_df,
    set_groupers,
    split_groups,
)

__all__ = [
    "groupbykey",
    "merge_df",
    "set_groupers",
    "split_groups",
    "UserDefinedGroupers",
]
