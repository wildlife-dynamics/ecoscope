from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from ecoscope.platform.tasks.preprocessing import (
    TrajectorySegmentFilter,
    process_relocations,
    relocations_to_trajectory,
)
from ecoscope.platform.tasks.transformation._filtering import Coordinate
from pydantic import ValidationError


def test_process_relocations():
    example_input_df_path = files("ecoscope.platform.tasks.io") / "get-subjectgroup-observations.example-return.parquet"
    input_df = gpd.read_parquet(example_input_df_path)
    kws = dict(
        filter_point_coords=[Coordinate(x=180, y=90), Coordinate(x=0, y=0)],
        relocs_columns=["groupby_col", "fixtime", "junk_status", "geometry"],
    )
    result = process_relocations(input_df, **kws)

    assert hasattr(result, "geometry")

    # we've cached this result for reuse by other tests, so check that cache is not stale
    cached_result_path = files("ecoscope.platform.tasks.preprocessing") / "process-relocations.example-return.parquet"
    cached = gpd.read_parquet(cached_result_path)
    pd.testing.assert_frame_equal(result, cached)


def test_relocations_to_trajectory():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "process-relocations.example-return.parquet"
    )
    input_df = gpd.read_parquet(example_input_df_path)

    traj_seg_filter = TrajectorySegmentFilter(
        min_length_meters=0.001,
        max_length_meters=10000,
        min_time_secs=1,
        max_time_secs=21600,
        min_speed_kmhr=0.01,
        max_speed_kmhr=10,
    )

    result = relocations_to_trajectory(input_df, trajectory_segment_filter=traj_seg_filter)

    assert hasattr(result, "geometry")

    # we've cached this result for reuse by other tests, so check that cache is not stale
    cached_result_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    cached = gpd.read_parquet(cached_result_path)
    pd.testing.assert_frame_equal(result, cached)


def test_relocations_to_trajectory_filter_clears_all():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "process-relocations.example-return.parquet"
    )
    input_df = gpd.read_parquet(example_input_df_path)

    trajectory_segment_filter = TrajectorySegmentFilter(
        min_length_meters=0.001,
        max_length_meters=0.002,
        min_time_secs=1,
        max_time_secs=1.001,
        min_speed_kmhr=0.01,
        max_speed_kmhr=0.02,
    )

    with pytest.raises(ValueError, match="No Trajectory data left after applying segment filter"):
        relocations_to_trajectory(input_df, trajectory_segment_filter=trajectory_segment_filter)


def test_traj_segment_filter_minimum_values():
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0.001"):
        TrajectorySegmentFilter(
            min_length_meters=0.0009,
        )
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
        TrajectorySegmentFilter(
            min_time_secs=0.9,
        )
    with pytest.raises(ValidationError, match="Input should be greater than 0.001"):
        TrajectorySegmentFilter(
            min_speed_kmhr=0.001,
        )


def test_traj_segment_filter_maximum_values():
    with pytest.raises(ValidationError, match="Input should be greater than 0.001"):
        TrajectorySegmentFilter(
            max_length_meters=0.001,
        )
    with pytest.raises(ValidationError, match="Input should be greater than 1"):
        TrajectorySegmentFilter(
            max_time_secs=1,
        )
    with pytest.raises(ValidationError, match="Input should be greater than 0.001"):
        TrajectorySegmentFilter(
            max_speed_kmhr=0.001,
        )


def test_traj_segment_filter_maximum_greater_than_minimum():
    with pytest.raises(
        (ValidationError, ValidationError),
        match="max_length_meters must be greater than min_length_meters",
    ):
        TrajectorySegmentFilter(
            max_length_meters=0.1,
            min_length_meters=25,
        )
        TrajectorySegmentFilter(
            max_length_meters=3,
            min_length_meters=3,
        )

    with pytest.raises(
        (ValidationError, ValidationError),
        match="max_time_secs must be greater than min_time_secs",
    ):
        TrajectorySegmentFilter(
            max_time_secs=25,
            min_time_secs=172800,
        )
        TrajectorySegmentFilter(
            max_time_secs=5,
            min_time_secs=5,
        )

    with pytest.raises(
        (ValidationError, ValidationError),
        match="max_speed_kmhr must be greater than min_speed_kmhr",
    ):
        TrajectorySegmentFilter(
            max_speed_kmhr=143,
            min_speed_kmhr=167,
        )
        TrajectorySegmentFilter(
            max_speed_kmhr=6,
            min_speed_kmhr=7,
        )
