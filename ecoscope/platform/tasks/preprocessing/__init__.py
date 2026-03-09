"""Relocation and trajectory preprocessing tasks.

Use this module to clean raw GPS relocations (removing outliers and redundant
fixes) and convert them into trajectory segments with speed, distance, and
heading attributes.
"""

from ._preprocessing import (
    TrajectorySegmentFilter,
    process_relocations,
    relocations_to_trajectory,
)

__all__ = [
    "process_relocations",
    "relocations_to_trajectory",
    "TrajectorySegmentFilter",
]
