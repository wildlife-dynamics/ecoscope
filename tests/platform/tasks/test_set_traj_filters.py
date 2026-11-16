import pytest
from wt_registry import get_registry

from ecoscope.platform.tasks.config import (
    get_bounding_box,
    get_filter_point_coords,
    get_segment_filter,
    get_segment_filter_overrides,
    set_traj_filters,
)
from ecoscope.platform.tasks.preprocessing._preprocessing import (
    TrajectorySegmentFilter,
    TrajectorySegmentFilterOverride,
)
from ecoscope.platform.tasks.transformation._filtering import (
    BoundingBox,
    Coordinate,
)


def _filters():
    return (
        BoundingBox(min_x=-10.0, max_x=20.0, min_y=-5.0, max_y=15.0),
        [Coordinate(x=10.0, y=20.0), Coordinate(x=-30.0, y=15.0)],
        TrajectorySegmentFilter(max_length_meters=50000),
        [TrajectorySegmentFilterOverride(key="elephant", filter=TrajectorySegmentFilter(max_speed_kmhr=12))],
    )


def test_set_traj_filters_returns_inputs_unchanged():
    bounding_box, filter_point_coords, trajectory_segment_filter, overrides = _filters()

    result = set_traj_filters(bounding_box, filter_point_coords, trajectory_segment_filter, overrides)

    assert result == (
        bounding_box,
        filter_point_coords,
        trajectory_segment_filter,
        overrides,
    )
    assert result[0] is bounding_box
    assert result[1] is filter_point_coords
    assert result[2] is trajectory_segment_filter
    assert result[3] is overrides


def test_set_traj_filters_none_coords_returns_default():
    result = set_traj_filters(filter_point_coords=None)

    assert result[1] == [
        Coordinate(x=180, y=90),
        Coordinate(x=0, y=0),
        Coordinate(x=1, y=1),
    ]


def test_get_bounding_box_returns_first_element():
    filters = _filters()

    assert get_bounding_box(filters) is filters[0]


def test_get_filter_point_coords_returns_second_element():
    filters = _filters()

    assert get_filter_point_coords(filters) is filters[1]


def test_get_segment_filter_returns_third_element():
    filters = _filters()

    assert get_segment_filter(filters) is filters[2]


def test_get_segment_filter_overrides_returns_fourth_element():
    filters = _filters()

    assert get_segment_filter_overrides(filters) is filters[3]


def test_overrides_default_to_undecided():
    assert set_traj_filters()[3] is None


def test_overrides_schema_matches_relocations_to_trajectory():
    registry = get_registry()
    set_filters = registry["ecoscope.platform.tasks.config._set_traj_filters.set_traj_filters"].json_schema
    to_traj = registry["ecoscope.platform.tasks.preprocessing._preprocessing.relocations_to_trajectory"].json_schema

    name = "trajectory_segment_filter_overrides"
    assert set_filters["properties"][name] == to_traj["properties"][name]
    assert (
        set_filters["$defs"]["TrajectorySegmentFilterOverride"] == to_traj["$defs"]["TrajectorySegmentFilterOverride"]
    )


@pytest.mark.parametrize(
    "name",
    ["set_traj_filters", "get_segment_filter", "get_segment_filter_overrides"],
)
def test_traj_filter_tasks_are_cacheable(name):
    entry = get_registry()[f"ecoscope.platform.tasks.config._set_traj_filters.{name}"]
    assert entry.declares_codec, entry.cache_declaration_error
