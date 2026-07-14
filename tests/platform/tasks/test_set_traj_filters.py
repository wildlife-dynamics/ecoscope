from ecoscope.platform.tasks.config import (
    get_bounding_box,
    get_filter_point_coords,
    get_segment_filter,
    set_traj_filters,
)
from ecoscope.platform.tasks.preprocessing._preprocessing import (
    TrajectorySegmentFilter,
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
    )


def test_set_traj_filters_returns_inputs_unchanged():
    bounding_box, filter_point_coords, trajectory_segment_filter = _filters()

    result = set_traj_filters(bounding_box, filter_point_coords, trajectory_segment_filter)

    assert result == (bounding_box, filter_point_coords, trajectory_segment_filter)
    assert result[0] is bounding_box
    assert result[1] is filter_point_coords
    assert result[2] is trajectory_segment_filter


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
