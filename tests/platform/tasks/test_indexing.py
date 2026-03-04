from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from ecoscope.platform.indexes import (
    AllGrouper,
    Month,
    SpatialGrouper,
    TemporalGrouper,
    ValueGrouper,
)
from ecoscope.platform.tasks.transformation import (
    add_spatial_index,
    add_temporal_index,
    resolve_spatial_feature_groups_for_spatial_groupers,
)

# --- Temporal indexing tests (from core) ---


def test_add_temporal_index():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-02-01", "2022-02-02"]),
            "value": [1, 2, 3, 4],
        }
    )
    assert list(df.index.values) == [0, 1, 2, 3]

    with_month_index = add_temporal_index(
        df,
        time_col="timestamp",
        groupers=[
            TemporalGrouper(
                temporal_index=Month(),
            ),
        ],
    )
    assert list(with_month_index.index.values) == [
        (0, "January"),
        (1, "January"),
        (2, "February"),
        (3, "February"),
    ]


# --- Spatial indexing tests (from ext) ---


@pytest.fixture
def example_trajectory():
    path = files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    return gpd.read_parquet(path)


@pytest.fixture
def example_events():
    path = files("ecoscope.platform.tasks.io") / "get-events.example-return.parquet"
    return gpd.read_parquet(path)


@pytest.fixture
def example_regions():
    path = files("ecoscope.platform.tasks.io") / "get-spatial-features-group.example-return.parquet"
    return gpd.read_parquet(path)


def test_add_spatial_index_traj(example_trajectory, example_regions):
    sg = SpatialGrouper(spatial_index_name="Test")
    sg.resolve(example_regions)

    result = add_spatial_index(gdf=example_trajectory, groupers=[sg])

    assert not result.empty
    assert sg.index_name in result.index.names
    region_names = set(example_regions["name"].tolist())
    region_pks = set(example_regions["pk"].tolist())
    valid_index_values = region_names | region_pks
    actual_index_values = set(result.index.get_level_values(sg.index_name))
    assert actual_index_values.issubset(valid_index_values)


def test_add_spatial_index_events(example_events, example_regions):
    sg = SpatialGrouper(spatial_index_name="Test")
    sg.resolve(example_regions)

    result = add_spatial_index(gdf=example_events, groupers=[sg])

    assert not result.empty
    assert sg.index_name in result.index.names
    assert len(result) <= len(example_events)
    region_names = set(example_regions["name"].tolist())
    region_pks = set(example_regions["pk"].tolist())
    valid_index_values = region_names | region_pks
    actual_index_values = set(result.index.get_level_values(sg.index_name))
    assert actual_index_values.issubset(valid_index_values)


def test_add_spatial_index_uses_pk_when_name_empty():
    regions = gpd.GeoDataFrame(
        {
            "pk": ["uuid-1", "uuid-2"],
            "name": ["", ""],
            "short_name": ["", ""],
            "feature_type": ["polygon", "polygon"],
            "metadata": [
                {"id": "group-1", "display_name": "Test"},
                {"id": "group-1", "display_name": "Test"},
            ],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        crs="EPSG:4326",
    )

    events = gpd.GeoDataFrame(
        {
            "id": ["e1", "e2"],
            "time": ["2024-01-01", "2024-01-02"],
            "event_type": ["type1", "type2"],
            "event_category": ["cat1", "cat2"],
            "reported_by": [None, None],
            "serial_number": [1, 2],
            "event_type_display": ["Type 1", "Type 2"],
        },
        geometry=[Point(0.5, 0.5), Point(1.5, 0.5)],
        crs="EPSG:4326",
    )

    sg = SpatialGrouper(spatial_index_name="Test")
    sg.resolve(regions)

    result = add_spatial_index(gdf=events, groupers=[sg])

    index_values = set(result.index.get_level_values(sg.index_name))
    # Should use uuid values since names are empty
    assert index_values == {"uuid-1", "uuid-2"}


def test_add_spatial_index_inner_join_drops_non_intersecting():
    regions = gpd.GeoDataFrame(
        {
            "pk": ["uuid-1"],
            "name": ["Region A"],
            "short_name": ["A"],
            "feature_type": ["polygon"],
            "metadata": [{"id": "group-1", "display_name": "Test"}],
        },
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )

    events = gpd.GeoDataFrame(
        {
            "id": ["inside", "outside"],
            "time": ["2024-01-01", "2024-01-02"],
            "event_type": ["type1", "type2"],
            "event_category": ["cat1", "cat2"],
            "reported_by": [None, None],
            "serial_number": [1, 2],
            "event_type_display": ["Type 1", "Type 2"],
        },
        geometry=[
            Point(0.5, 0.5),
            Point(10, 10),
        ],
        crs="EPSG:4326",
    )

    sg = SpatialGrouper(spatial_index_name="Test")
    sg.resolve(regions)

    result = add_spatial_index(gdf=events, groupers=[sg])

    assert len(result) == 1
    assert "inside" in result["id"].values


def test_add_spatial_index_overlapping_regions_duplicates_rows():
    """
    Verify that events in overlapping regions appear multiple times.
    This is expected behavior that may or may not be expected from a user input POV.
    """
    regions = gpd.GeoDataFrame(
        {
            "pk": ["uuid-1", "uuid-2"],
            "name": ["Region A", "Region B"],
            "short_name": ["A", "B"],
            "feature_type": ["polygon", "polygon"],
            "metadata": [
                {"id": "group-1", "display_name": "Test"},
                {"id": "group-1", "display_name": "Test"},
            ],
        },
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # Larger region
            Polygon([(0.25, 0.25), (1.75, 0.25), (1.75, 1.75), (0.25, 1.75)]),  # Overlapping
        ],
        crs="EPSG:4326",
    )

    events = gpd.GeoDataFrame(
        {
            "id": ["in-both"],
            "time": ["2024-01-01"],
            "event_type": ["type1"],
            "event_category": ["cat1"],
            "reported_by": [None],
            "serial_number": [1],
            "event_type_display": ["Type 1"],
        },
        geometry=[Point(1, 1)],  # Inside both regions
        crs="EPSG:4326",
    )

    sg = SpatialGrouper(spatial_index_name="Test")
    sg.resolve(regions)

    result = add_spatial_index(gdf=events, groupers=[sg])

    # Event should appear twice, once for each region
    assert len(result) == 2
    index_values = set(result.index.get_level_values(sg.index_name))
    assert index_values == {"Region A", "Region B"}


def test_add_spatial_index_filters_non_polygon_geometries():
    # Create regions with mixed geometry types
    regions = gpd.GeoDataFrame(
        {
            "pk": ["poly-1", "line-1", "point-1"],
            "name": ["Polygon Region", "Line Region", "Point Region"],
            "short_name": ["P", "L", "Pt"],
            "feature_type": ["polygon", "line", "point"],
            "metadata": [
                {"id": "group-1", "display_name": "Test"},
                {"id": "group-1", "display_name": "Test"},
                {"id": "group-1", "display_name": "Test"},
            ],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            LineString([(2, 0), (3, 1)]),
            Point(4, 4),
        ],
        crs="EPSG:4326",
    )

    sg = SpatialGrouper(spatial_index_name="Test")
    groupers = resolve_spatial_feature_groups_for_spatial_groupers(
        groupers=[sg],
        spatial_feature_groups=regions,
    )

    # The resolved spatial regions should only contain the polygon
    resolved_sg = groupers[0]
    assert len(resolved_sg.spatial_regions) == 1
    assert resolved_sg.spatial_regions.iloc[0]["name"] == "Polygon Region"


def test_add_spatial_index_filters_all_polygon_geometries_raises():
    regions = gpd.GeoDataFrame(
        {
            "pk": ["uuid-1", "uuid-2"],
            "name": ["", ""],
            "short_name": ["", ""],
            "feature_type": ["polygon", "polygon"],
            "metadata": [
                {"id": "group-1", "display_name": "Test"},
                {"id": "group-1", "display_name": "Test"},
            ],
        },
        geometry=[
            Point(0, 0),
            Point(1, 1),
        ],
        crs="EPSG:4326",
    )
    sg = SpatialGrouper(spatial_index_name="Test")

    with pytest.raises(
        ValueError,
        match=(
            "There are no polygons in Feature Group Test,"
            " you must select a feature collection that contains at least 1 polygon."
        ),
    ):
        resolve_spatial_feature_groups_for_spatial_groupers(
            groupers=[sg],
            spatial_feature_groups=regions,
        )


def test_add_spatial_index_unresolved_grouper_passthrough():
    events = gpd.GeoDataFrame(
        {
            "id": ["e1"],
            "time": ["2024-01-01"],
            "event_type": ["type1"],
            "event_category": ["cat1"],
            "reported_by": [None],
            "serial_number": [1],
            "event_type_display": ["Type 1"],
        },
        geometry=[Point(0.5, 0.5)],
        crs="EPSG:4326",
    )

    sg = SpatialGrouper(spatial_index_name="unresolved-name")
    result = add_spatial_index(gdf=events, groupers=[sg])
    pd.testing.assert_frame_equal(result, events)


def test_add_spatial_index_all_grouper_passthrough(example_events):
    result = add_spatial_index(gdf=example_events, groupers=AllGrouper())
    pd.testing.assert_frame_equal(result, example_events)


def test_add_spatial_index_mixed_groupers_ignores_non_spatial(example_events, example_regions):
    sg = SpatialGrouper(spatial_index_name="My Features")
    sg.resolve(example_regions)
    vg = ValueGrouper(index_name="event_type")
    tg = TemporalGrouper(temporal_index=Month())

    result = add_spatial_index(gdf=example_events, groupers=[sg, vg, tg])

    # Only spatial grouper index should be added
    assert sg.index_name in result.index.names
    assert vg.index_name not in result.index.names
    assert tg.index_name not in result.index.names


def test_add_spatial_index_multiple_spatial_groupers():
    regions1 = gpd.GeoDataFrame(
        {
            "pk": ["r1-uuid"],
            "name": ["Global Feature"],
            "short_name": ["G"],
            "feature_type": ["polygon"],
            "metadata": [{"id": "group-1", "display_name": "Group 1"}],
        },
        geometry=[Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])],
        crs="EPSG:4326",
    )

    regions2 = gpd.GeoDataFrame(
        {
            "pk": ["l1-uuid", "l2-uuid"],
            "name": ["Local Feature 1", "Local Feature 2"],
            "short_name": ["L1", "L2"],
            "feature_type": ["polygon", "polygon"],
            "metadata": [
                {"id": "group-2", "display_name": "Group 2"},
                {"id": "group-2", "display_name": "Group 2"},
            ],
        },
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(2, 4), (4, 2), (4, 4), (2, 4)]),
        ],
        crs="EPSG:4326",
    )

    events = gpd.GeoDataFrame(
        {
            "id": ["e1"],
            "time": ["2024-01-01"],
            "event_type": ["type1"],
            "event_category": ["cat1"],
            "reported_by": [None],
            "serial_number": [1],
            "event_type_display": ["Type 1"],
        },
        geometry=[Point(1, 1)],
        crs="EPSG:4326",
    )

    sg1 = SpatialGrouper(spatial_index_name="Group 1")
    sg1.resolve(regions1)
    sg2 = SpatialGrouper(spatial_index_name="Group 2")
    sg2.resolve(regions2)

    result = add_spatial_index(gdf=events, groupers=[sg1, sg2])

    # Both spatial grouper indexes should be present
    assert sg1.index_name in result.index.names
    assert sg2.index_name in result.index.names
    # E1 should be in Local Feature 1
    assert result.loc[(0, "Global Feature", "Local Feature 1")]["id"] == "e1"
