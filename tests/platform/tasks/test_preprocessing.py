import datetime
from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
import shapely
from pydantic import TypeAdapter, ValidationError
from wt_registry import get_registry

from ecoscope.platform.tasks.preprocessing import (
    TrajectorySegmentFilter,
    TrajectorySegmentFilterOverride,
    TrajectorySegmentFilterOverrides,
    convert_trajectory_to_relocations,
    process_relocations,
    relocations_to_trajectory,
)
from ecoscope.platform.tasks.transformation._filtering import Coordinate
from ecoscope.trajectory import TrajSegFilter

# Degrees of longitude stepped per hourly fix, keyed by the test subject id below;
# at the equator this gives ~1.1 km/h ("ele") and ~111 km/h ("heli").
_LON_STEP_DEG = {"ele": 0.01, "heli": 1.0}


def _relocs(subtypes: dict[str, str | None]) -> gpd.GeoDataFrame:
    """Build a SubjectGroupObservationsGDF-shaped frame: one subject per key, 5 hourly fixes each."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [
        {
            "groupby_col": subject_id,
            "fixtime": start + datetime.timedelta(hours=i),
            "junk_status": False,
            "subject_subtype": subtype,
            "geometry": shapely.Point(_LON_STEP_DEG[subject_id] * i, 0.0),
        }
        for subject_id, subtype in subtypes.items()
        for i in range(5)
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


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


def test_preset_is_accepted_and_omitted_from_schema():
    f = TrajectorySegmentFilter(preset="elephant", max_speed_kmhr=12)
    assert f.preset == "elephant"
    assert "preset" not in TrajectorySegmentFilter.model_json_schema()["properties"]


def test_preset_is_ignored_by_task():
    core = TrajectorySegmentFilter(preset="elephant", max_speed_kmhr=12).to_traj_seg_filter()
    assert core == TrajSegFilter(
        min_length_meters=0.001,
        max_length_meters=100000,
        min_time_secs=1,
        max_time_secs=172800,
        min_speed_kmhr=0.01,
        max_speed_kmhr=12,
    )


def test_duplicate_override_keys_rejected():
    ta = TypeAdapter(TrajectorySegmentFilterOverrides)
    row = {"key": "elephant", "filter": {}}
    with pytest.raises(ValidationError, match="Duplicate .* elephant"):
        ta.validate_python([row, row])


def test_overrides_validate_from_dicts():
    # Exercises the pydantic validation path the server's dict payload goes through;
    # `relocations_to_trajectory` itself is called directly in tests below (unvalidated,
    # like the other task tests in this file), so construct the models explicitly there.
    ta = TypeAdapter(TrajectorySegmentFilterOverrides)
    parsed = ta.validate_python([{"key": "helicopter", "filter": {"max_speed_kmhr": 300}}])
    assert parsed == [
        TrajectorySegmentFilterOverride(key="helicopter", filter=TrajectorySegmentFilter(max_speed_kmhr=300))
    ]


def test_overrides_apply_per_kind():
    default = TrajectorySegmentFilter(max_speed_kmhr=50)
    # heli's ~111km/hour segments also exceed the 100km default max_length_meters, so its
    # override must raise that bound too, or the length check would mask the speed override.
    overrides = [
        TrajectorySegmentFilterOverride(
            key="helicopter", filter=TrajectorySegmentFilter(max_speed_kmhr=300, max_length_meters=150000)
        )
    ]
    out = relocations_to_trajectory(
        _relocs({"ele": "elephant", "heli": "helicopter"}), default, overrides, "subject_subtype"
    )
    assert set(out["groupby_col"]) == {"ele", "heli"}  # heli kept by its override


def test_overrides_drop_only_their_kind():
    default = TrajectorySegmentFilter(max_speed_kmhr=300, max_length_meters=150000)
    overrides = [TrajectorySegmentFilterOverride(key="elephant", filter=TrajectorySegmentFilter(max_speed_kmhr=0.5))]
    out = relocations_to_trajectory(
        _relocs({"ele": "elephant", "heli": "helicopter"}), default, overrides, "subject_subtype"
    )
    assert set(out["groupby_col"]) == {"heli"}


def test_overrides_nan_key_uses_default():
    default = TrajectorySegmentFilter(max_speed_kmhr=50)
    overrides = [TrajectorySegmentFilterOverride(key="helicopter", filter=TrajectorySegmentFilter(max_speed_kmhr=300))]
    out = relocations_to_trajectory(_relocs({"ele": None, "heli": None}), default, overrides, "subject_subtype")
    assert set(out["groupby_col"]) == {"ele"}  # heli has no key -> default drops it


def test_overrides_categorical_key_column():
    relocs = _relocs({"ele": "elephant", "heli": "helicopter"})
    relocs["subject_subtype"] = relocs["subject_subtype"].astype("category")
    out = relocations_to_trajectory(
        relocs,
        TrajectorySegmentFilter(max_speed_kmhr=50),
        [
            TrajectorySegmentFilterOverride(
                key="helicopter", filter=TrajectorySegmentFilter(max_speed_kmhr=300, max_length_meters=150000)
            )
        ],
        "subject_subtype",
    )
    assert set(out["groupby_col"]) == {"ele", "heli"}


def test_overrides_without_column_raises():
    with pytest.raises(ValueError, match="Filter overrides aren't supported by this workflow"):
        relocations_to_trajectory(
            _relocs({"ele": "elephant"}),
            None,
            [TrajectorySegmentFilterOverride(key="elephant", filter=TrajectorySegmentFilter())],
            None,
        )


def test_overrides_missing_column_raises():
    with pytest.raises(ValueError, match="patrol_type__value"):
        relocations_to_trajectory(
            _relocs({"ele": "elephant"}),
            None,
            [TrajectorySegmentFilterOverride(key="elephant", filter=TrajectorySegmentFilter())],
            "patrol_type__value",
        )


def test_overrides_keyed_on_core_trajectory_column():
    # `groupby_col` is a core trajectory column, not an `extra__`-prefixed relocation column
    # (`_create_trajsegments` drops the extra copy rather than prefixing it) - exercises the
    # bare-name resolution path, keying directly on subject id.
    default = TrajectorySegmentFilter(max_speed_kmhr=50)
    overrides = [
        TrajectorySegmentFilterOverride(
            key="heli", filter=TrajectorySegmentFilter(max_speed_kmhr=300, max_length_meters=150000)
        )
    ]
    out = relocations_to_trajectory(
        _relocs({"ele": "elephant", "heli": "helicopter"}), default, overrides, "groupby_col"
    )
    assert set(out["groupby_col"]) == {"ele", "heli"}


def test_overrides_unmatched_key_matches_no_overrides():
    relocs = _relocs({"ele": "elephant", "heli": "helicopter"})
    f = TrajectorySegmentFilter(max_speed_kmhr=50)
    overrides = [TrajectorySegmentFilterOverride(key="zebra", filter=TrajectorySegmentFilter(max_speed_kmhr=300))]
    pd.testing.assert_frame_equal(
        relocations_to_trajectory(relocs, f, overrides, "subject_subtype"),
        relocations_to_trajectory(relocs, f),
    )


def test_overrides_schema():
    # Regression guard: see `trajectory_segment_filter_override_column`'s annotation comment.
    schema = get_registry()[
        "ecoscope.platform.tasks.preprocessing._preprocessing.relocations_to_trajectory"
    ].json_schema
    props = schema["properties"]
    assert props["trajectory_segment_filter_overrides"]["ecoscope:advanced"] is True
    assert props["trajectory_segment_filter_overrides"]["items"] == {"$ref": "#/$defs/TrajectorySegmentFilterOverride"}
    assert "trajectory_segment_filter_override_column" not in props
    assert "preset" not in schema["$defs"]["TrajectorySegmentFilter"]["properties"]


def test_preset_omitted_from_serialized_defaults():
    # `SkipJsonSchema` only hides `preset` from $defs; the field-level `default` for
    # `trajectory_segment_filter` is separately serialized from a `TrajectorySegmentFilter()`
    # instance, so it needs its own guard (the `model_serializer` on the class).
    from ecoscope.platform.tasks.config import set_traj_filters  # noqa: F401  (registers it)

    relocs_schema = get_registry()[
        "ecoscope.platform.tasks.preprocessing._preprocessing.relocations_to_trajectory"
    ].json_schema
    set_filters_schema = get_registry()["ecoscope.platform.tasks.config._set_traj_filters.set_traj_filters"].json_schema

    assert "preset" not in relocs_schema["properties"]["trajectory_segment_filter"]["default"]
    assert "preset" not in set_filters_schema["properties"]["trajectory_segment_filter"]["default"]
    assert TrajectorySegmentFilter(preset="elephant").model_dump()["preset"] == "elephant"


def test_empty_overrides_match_no_overrides():
    relocs = _relocs({"ele": "elephant", "heli": "helicopter"})
    f = TrajectorySegmentFilter(max_speed_kmhr=50)
    pd.testing.assert_frame_equal(
        relocations_to_trajectory(relocs, f, [], "subject_subtype"),
        relocations_to_trajectory(relocs, f),
    )


def test_convert_trajectory_to_relocations():
    from ecoscope.trajectory import Trajectory

    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    trajectory_gdf = gpd.read_parquet(example_input_df_path)

    result = convert_trajectory_to_relocations(trajectory_gdf)

    assert (result.geom_type == "Point").all()
    assert len(result) > len(trajectory_gdf)  # more fixes recovered than segments (fixes minus one per subject)

    expected = Trajectory(gdf=trajectory_gdf).to_relocations().gdf
    pd.testing.assert_frame_equal(result, expected)
