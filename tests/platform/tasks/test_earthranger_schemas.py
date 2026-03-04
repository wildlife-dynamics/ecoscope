import uuid

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from ecoscope.platform.schemas import (
    EventGDF,
    EventsWithDisplayNamesGDF,
    PatrolObservationsGDF,
    PatrolsDF,
    RegionsGDF,
    RelocationsGDFSchema,
    SubjectGroupObservationsGDF,
    TrajectoryGDF,
)
from pydantic import TypeAdapter, ValidationError
from shapely.geometry import LineString, Point


def test_subjectgroupobservations_schema():
    ta = TypeAdapter(SubjectGroupObservationsGDF)

    gdf_without_null = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "junk_status": [False, False],
            "extra__subject__name": ["Test", float("nan")],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )

    result = ta.validate_python(gdf_without_null)
    allowed_missing_columns = [
        "extra__subject__name",
        "extra__subject__subject_subtype",
        "extra__subject__sex",
    ]
    assert [col in result for col in allowed_missing_columns]
    assert result.extra__subject__name.to_list() == ["Test", "None"]
    assert result.extra__subject__subject_subtype.to_list() == ["None", "None"]
    assert result.extra__subject__sex.to_list() == ["None", "None"]

    gdf_with_null = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"]),
            "junk_status": [False, False],
            "geometry": [Point(0.0, 0.0), None],
        }
    )
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_null)

    gdf_with_missing_column = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            # "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"]),
            "junk_status": [False, False],
            "geometry": [Point(0.0, 0.0), None],
        }
    )
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_missing_column)

    gdf_with_naive_timestamp = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=False),
            "junk_status": [False, False],
            "extra__subject__name": ["Test", float("nan")],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )

    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_naive_timestamp)


def test_events_schema():
    ta = TypeAdapter(EventGDF)

    gdf_without_null = gpd.GeoDataFrame(
        {
            "id": ["A", "B"],
            "event_type": ["Type1", "Type2"],
            "time": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
            "reported_by": [None, {"name": "some_subject"}],
        }
    )

    result = ta.validate_python(gdf_without_null)
    allowed_missing_columns = [
        "event_category",
        "reported_by",
    ]
    assert [col in result for col in allowed_missing_columns]
    assert result.event_category.to_list() == ["None", "None"]
    assert result.reported_by.to_list() == [
        {"name": "None"},
        {"name": "some_subject"},
    ]

    gdf_with_null = gpd.GeoDataFrame(
        {
            "id": ["A", "B"],
            "event_type": ["Type1", "Type2"],
            "time": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "geometry": [Point(0.0, 0.0), None],
        }
    )
    # This should pass since we allow null geometry in events
    ta.validate_python(gdf_with_null)

    gdf_with_missing_column = gpd.GeoDataFrame(
        {
            "id": ["A", "B"],
            # "event_type": ["Type1", "Type2"],
            "time": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "geometry": [Point(0.0, 0.0), None],
        }
    )
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_missing_column)

    gdf_with_naive_timestamp = gpd.GeoDataFrame(
        {
            "id": ["A", "B"],
            "event_type": ["Type1", "Type2"],
            "time": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=False),
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
            "reported_by": [None, {"name": "some_subject"}],
        }
    )

    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_naive_timestamp)


def test_patrol_obs_schema():
    ta = TypeAdapter(PatrolObservationsGDF)

    gdf = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "junk_status": [False, False],
            "patrol_serial_number": [1234, float("nan")],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )
    result = ta.validate_python(gdf)
    allowed_missing_columns = [
        "patrol_type__value",
        "patrol_serial_number",
        "patrol_status",
        "patrol_subject",
    ]
    assert [col in result for col in allowed_missing_columns]
    assert result.patrol_type__value.to_list() == ["None", "None"]
    assert result.patrol_serial_number.to_list() == [
        "1234",
        "None",
    ]  # Ensure type coercion on this column explicitly
    assert result.patrol_status.to_list() == ["None", "None"]
    assert result.patrol_subject.to_list() == ["None", "None"]

    gdf_with_null = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "junk_status": [False, False],
            "extra__patrol_type__value": ["Test", float("nan")],
            "geometry": [Point(0.0, 0.0), None],
        }
    )
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_null)

    gdf_with_missing_column = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            # "junk_status": [False, False],
            "extra__patrol_type__value": ["Test", float("nan")],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_missing_column)

    gdf_with_naive_timestamp = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=False),
            "junk_status": [False, False],
            "extra__subject__name": ["Test", float("nan")],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )

    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_naive_timestamp)


def test_patrol_and_subject_obs_are_relocs():
    subject_obs = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "junk_status": [False, False],
            "extra__subject__name": ["Test", float("nan")],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )
    validated_subject_obs = TypeAdapter(SubjectGroupObservationsGDF).validate_python(
        subject_obs
    )

    patrol_obs = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "junk_status": [False, False],
            "extra__subject__name": ["Test", float("nan")],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )
    validated_patrol_obs = TypeAdapter(SubjectGroupObservationsGDF).validate_python(
        patrol_obs
    )

    RelocationsGDFSchema.validate(validated_subject_obs)
    RelocationsGDFSchema.validate(validated_patrol_obs)


def test_patrols_schema():
    ta = TypeAdapter(PatrolsDF)
    df = pd.DataFrame(
        {
            "id": ["1234", "5678"],
            "state": ["done", "done"],
            "serial_number": [1234, 4567],
            "patrol_segments": [{}, {}],
        }
    )
    ta.validate_python(df)

    df_with_null = gpd.GeoDataFrame(
        {
            "id": ["1234", "5678"],
            "state": ["done", None],
            "serial_number": [1234, 4567],
            "patrol_segments": [{}, {}],
        }
    )
    with pytest.raises(ValidationError):
        ta.validate_python(df_with_null)

    df_with_missing_column = gpd.GeoDataFrame(
        {
            "id": ["1234", "5678"],
            "state": ["done", "done"],
            # "serial_number": [1234, 4567],
            "patrol_segments": [{}, {}],
        }
    )
    with pytest.raises(ValidationError):
        ta.validate_python(df_with_missing_column)


def test_trajectory_schema():
    ta = TypeAdapter(TrajectoryGDF)

    gdf_without_null = gpd.GeoDataFrame(
        index=pd.Index(data=["A", "B"], name="id"),
        data={
            "groupby_col": ["Type1", "Type2"],
            "segment_start": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "segment_end": pd.to_datetime(["2024-01-02", "2025-01-02"], utc=True),
            "timespan_seconds": [86400.0, 86400.0],
            "dist_meters": [12000.0, 15000.0],
            "speed_kmhr": [0.5, 0.625],
            "heading": [11.1, 23.4],
            "junk_status": [False, False],
            "geometry": [
                LineString([[0, 0], [1, 0], [1, 1]]),
                LineString([[10, 11], [11, 13], [15, 20]]),
            ],
        },
    )
    ta.validate_python(gdf_without_null)

    gdf_with_missing_column = gdf_without_null.drop(columns=["groupby_col"])
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_missing_column)

    gdf_with_naive_segment_start = gdf_without_null.copy()
    gdf_with_naive_segment_start["segment_start"] = gdf_with_naive_segment_start[
        "segment_start"
    ].apply(lambda x: x.replace(tzinfo=None))
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_naive_segment_start)

    gdf_with_naive_segment_end = gdf_without_null.copy()
    gdf_with_naive_segment_end["segment_end"] = gdf_with_naive_segment_end[
        "segment_start"
    ].apply(lambda x: x.replace(tzinfo=None))
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_naive_segment_end)


def test_events_schema_with_display_names():
    ta = TypeAdapter(EventsWithDisplayNamesGDF)

    event_gdf = gpd.GeoDataFrame(
        {
            "id": ["A", "B"],
            "event_type": ["Type1", "Type2"],
            "event_type_display": ["Type One", "Type Two"],
            "time": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
            "reported_by": [None, {"name": "some_subject"}],
        }
    )
    ta.validate_python(event_gdf)

    gdf_with_missing_column = event_gdf.drop(columns=["event_type_display"])
    with pytest.raises(ValidationError):
        ta.validate_python(gdf_with_missing_column)


def test_regions_schema():
    ta = TypeAdapter(RegionsGDF)

    regions = gpd.GeoDataFrame(
        {
            "pk": [str(uuid.uuid4()), str(uuid.uuid4())],
            "name": ["", "Feature 2"],
            "short_name": ["Feat 1", ""],
            "feature_type": [str(uuid.uuid4()), str(uuid.uuid4())],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
            "metadata": [
                {"id": "1234", "display_name": "group name"},
                {"id": "1234", "display_name": "group name"},
            ],
        }
    )
    ta.validate_python(regions)
