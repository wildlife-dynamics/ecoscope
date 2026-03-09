"""Warning helpers for runtime workflow diagnostics.

Provides functions that emit warnings when data quality issues are detected,
such as mixed subject subtypes within a single grouper split.
"""

from ._warning import (
    mixed_subtype_warning,
)

__all__ = [
    "mixed_subtype_warning",
]
