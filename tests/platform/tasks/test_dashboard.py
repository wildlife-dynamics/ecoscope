from datetime import datetime

import pytest
from ecoscope.platform.indexes import Month, TemporalGrouper, ValueGrouper, Year
from ecoscope.platform.tasks.config._workflow_details import WorkflowDetails
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.results import gather_dashboard
from ecoscope.platform.tasks.results._dashboard import (
    Dashboard,
    EmumeratedWidgetSingleView,
    Metadata,
)
from ecoscope.platform.tasks.results._widget_types import GroupedWidget

DashboardFixture = tuple[list[GroupedWidget], Dashboard]


def assert_dashboards_equal(d1: Dashboard, d2: Dashboard):
    assert d1.grouper_choices
    assert d2.grouper_choices
    assert d1.grouper_choices.keys() == d2.grouper_choices.keys()
    assert list(d1.grouper_choices.values()) == list(d2.grouper_choices.values())
    assert d1.keys == d2.keys
    assert d1.widgets == d2.widgets
    assert d1.metadata == d2.metadata


@pytest.fixture
def single_filter_dashboard() -> DashboardFixture:
    great_map = GroupedWidget(
        widget_type="map",
        title="A Great Map",
        views={
            (("TemporalGrouper_%B", "=", "January"),): "/path/to/precomputed/jan/map.html",
            (("TemporalGrouper_%B", "=", "February"),): "/path/to/precomputed/feb/map.html",
        },
        is_filtered=True,
    )
    cool_plot = GroupedWidget(
        widget_type="graph",
        title="A Cool Plot",
        views={
            (("TemporalGrouper_%B", "=", "January"),): "/path/to/precomputed/jan/plot.html",
            (("TemporalGrouper_%B", "=", "February"),): "/path/to/precomputed/feb/plot.html",
        },
        is_filtered=True,
    )
    widgets = [great_map, cool_plot]
    dashboard = Dashboard(
        grouper_choices={TemporalGrouper(temporal_index=Month()): ["January", "February"]},
        keys=[
            (("TemporalGrouper_%B", "=", "February"),),
            (("TemporalGrouper_%B", "=", "January"),),
        ],
        widgets=[great_map, cool_plot],
        metadata=Metadata(
            title="A Great Dashboard",
            description="A dashboard with a map and a plot",
            time_range="From 01 Jan 2011 00:00:00 to 01 Jan 2023 00:00:00",
            time_zone="UTC",
        ),
    )
    return widgets, dashboard


def test_gather_dashboard(single_filter_dashboard: DashboardFixture):
    grouped_widgets, expected_dashboard = single_filter_dashboard
    dashboard: Dashboard = gather_dashboard(
        details=WorkflowDetails(
            name="A Great Dashboard",
            description="A dashboard with a map and a plot",
        ),
        time_range=TimeRange(
            since=datetime.strptime("2011-01-01", "%Y-%m-%d"),
            until=datetime.strptime("2023-01-01", "%Y-%m-%d"),
            timezone=UTC_TIMEZONEINFO,
        ),
        widgets=grouped_widgets,
        groupers=[TemporalGrouper(temporal_index=Month())],
    )
    assert_dashboards_equal(dashboard, expected_dashboard)


def test__get_view(single_filter_dashboard: DashboardFixture):
    _, dashboard = single_filter_dashboard
    assert dashboard._get_view((("TemporalGrouper_%B", "=", "January"),)) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/precomputed/jan/map.html",
            is_filtered=True,
        ),
        EmumeratedWidgetSingleView(
            id=1,
            widget_type="graph",
            title="A Cool Plot",
            data="/path/to/precomputed/jan/plot.html",
            is_filtered=True,
        ),
    ]
    assert dashboard._get_view((("TemporalGrouper_%B", "=", "February"),)) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/precomputed/feb/map.html",
            is_filtered=True,
        ),
        EmumeratedWidgetSingleView(
            id=1,
            widget_type="graph",
            title="A Cool Plot",
            data="/path/to/precomputed/feb/plot.html",
            is_filtered=True,
        ),
    ]


def test_model_dump_views(single_filter_dashboard: DashboardFixture):
    _, dashboard = single_filter_dashboard
    assert dashboard.model_dump()["views"] == {
        '{"TemporalGrouper_%B": "January"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/precomputed/jan/map.html",
                "is_filtered": True,
            },
            {
                "id": 1,
                "widget_type": "graph",
                "title": "A Cool Plot",
                "data": "/path/to/precomputed/jan/plot.html",
                "is_filtered": True,
            },
        ],
        '{"TemporalGrouper_%B": "February"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/precomputed/feb/map.html",
                "is_filtered": True,
            },
            {
                "id": 1,
                "widget_type": "graph",
                "title": "A Cool Plot",
                "data": "/path/to/precomputed/feb/plot.html",
                "is_filtered": True,
            },
        ],
    }


def test_model_dump_filters(single_filter_dashboard: DashboardFixture):
    _, dashboard = single_filter_dashboard
    assert dashboard.model_dump()["filters"] == {
        "schema": {
            "type": "object",
            "properties": {
                "TemporalGrouper_%B": {
                    "type": "string",
                    "title": "Month",
                    "oneOf": [
                        {"const": "January", "title": "January"},
                        {"const": "February", "title": "February"},
                    ],
                    "default": "January",
                },
            },
            "uiSchema": {
                "TemporalGrouper_%B": {
                    "ui:title": "Month",
                    "ui:widget": "select",
                },
            },
        }
    }


@pytest.fixture
def two_filter_dashboard() -> DashboardFixture:
    great_map = GroupedWidget(
        widget_type="map",
        title="A Great Map",
        views={
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2022"),
            ): "/path/to/jan/2022/map.html",
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2023"),
            ): "/path/to/jan/2023/map.html",
        },
        is_filtered=True,
    )
    widgets = [great_map]
    dashboard = Dashboard(
        grouper_choices={
            TemporalGrouper(temporal_index=Month()): ["January"],
            TemporalGrouper(temporal_index=Year()): ["2022", "2023"],
        },
        keys=[
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2022"),
            ),
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2023"),
            ),
        ],
        widgets=widgets,
        metadata=Metadata(
            title="A Great Dashboard",
            description="A dashboard with a map",
            time_range="From 01 Jan 2011 00:00:00 to 01 Jan 2023 00:00:00",
            time_zone="UTC",
        ),
    )
    return widgets, dashboard


def test_gather_dashboard_two_filter(two_filter_dashboard: DashboardFixture):
    grouped_widgets, expected_dashboard = two_filter_dashboard
    dashboard: Dashboard = gather_dashboard(
        details=WorkflowDetails(
            name="A Great Dashboard",
            description="A dashboard with a map",
        ),
        time_range=TimeRange(
            since=datetime.strptime("2011-01-01", "%Y-%m-%d"),
            until=datetime.strptime("2023-01-01", "%Y-%m-%d"),
            timezone=UTC_TIMEZONEINFO,
        ),
        widgets=grouped_widgets,
        groupers=[
            TemporalGrouper(temporal_index=Month()),
            TemporalGrouper(temporal_index=Year()),
        ],
    )
    assert_dashboards_equal(dashboard, expected_dashboard)


def test__get_view_two_part_key(two_filter_dashboard: DashboardFixture):
    _, dashboard = two_filter_dashboard
    assert dashboard._get_view((("TemporalGrouper_%B", "=", "January"), ("TemporalGrouper_%Y", "=", "2022"))) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/jan/2022/map.html",
            is_filtered=True,
        ),
    ]
    assert dashboard._get_view((("TemporalGrouper_%B", "=", "January"), ("TemporalGrouper_%Y", "=", "2023"))) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/jan/2023/map.html",
            is_filtered=True,
        ),
    ]


def test_model_dump_views_two_filter(two_filter_dashboard: DashboardFixture):
    _, dashboard = two_filter_dashboard
    assert dashboard.model_dump()["views"] == {
        '{"TemporalGrouper_%B": "January", "TemporalGrouper_%Y": "2022"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/jan/2022/map.html",
                "is_filtered": True,
            },
        ],
        '{"TemporalGrouper_%B": "January", "TemporalGrouper_%Y": "2023"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/jan/2023/map.html",
                "is_filtered": True,
            },
        ],
    }


def test_model_dump_filters_two_filter(two_filter_dashboard: DashboardFixture):
    _, dashboard = two_filter_dashboard
    assert dashboard.model_dump()["filters"] == {
        "schema": {
            "type": "object",
            "properties": {
                "TemporalGrouper_%B": {
                    "type": "string",
                    "title": "Month",
                    "oneOf": [
                        {"const": "January", "title": "January"},
                    ],
                    "default": "January",
                },
                "TemporalGrouper_%Y": {
                    "type": "string",
                    "title": "Year",
                    "oneOf": [
                        {"const": "2022", "title": "2022"},
                        {"const": "2023", "title": "2023"},
                    ],
                    "default": "2022",
                },
            },
            "uiSchema": {
                "TemporalGrouper_%B": {
                    "ui:title": "Month",
                    "ui:widget": "select",
                },
                "TemporalGrouper_%Y": {
                    "ui:title": "Year",
                    "ui:widget": "select",
                },
            },
        }
    }


@pytest.fixture
def three_filter_dashboard() -> DashboardFixture:
    great_map = GroupedWidget(
        widget_type="map",
        title="A Great Map",
        views={
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2022"),
                ("subject_name", "=", "jo"),
            ): "/path/to/jan/2022/jo/map.html",
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2022"),
                ("subject_name", "=", "zo"),
            ): "/path/to/jan/2022/zo/map.html",
        },
        is_filtered=True,
    )
    widgets = [great_map]
    dashboard = Dashboard(
        grouper_choices={
            TemporalGrouper(temporal_index=Month()): ["January"],
            TemporalGrouper(temporal_index=Year()): ["2022"],
            ValueGrouper(index_name="subject_name"): ["jo", "zo"],
        },
        keys=[
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2022"),
                ("subject_name", "=", "jo"),
            ),
            (
                ("TemporalGrouper_%B", "=", "January"),
                ("TemporalGrouper_%Y", "=", "2022"),
                ("subject_name", "=", "zo"),
            ),
        ],
        widgets=widgets,
        metadata=Metadata(
            title="A Great Dashboard",
            description="A dashboard with a map",
            time_range="From 01 Jan 2011 00:00:00 to 01 Jan 2023 00:00:00",
            time_zone="UTC",
        ),
    )
    return widgets, dashboard


def test_gather_dashboard_three_filter(three_filter_dashboard: DashboardFixture):
    grouped_widgets, expected_dashboard = three_filter_dashboard
    dashboard: Dashboard = gather_dashboard(
        details=WorkflowDetails(
            name="A Great Dashboard",
            description="A dashboard with a map",
        ),
        time_range=TimeRange(
            since=datetime.strptime("2011-01-01", "%Y-%m-%d"),
            until=datetime.strptime("2023-01-01", "%Y-%m-%d"),
            timezone=UTC_TIMEZONEINFO,
        ),
        widgets=grouped_widgets,
        groupers=[
            TemporalGrouper(temporal_index=Month()),
            TemporalGrouper(temporal_index=Year()),
            ValueGrouper(index_name="subject_name"),
        ],
    )
    assert_dashboards_equal(dashboard, expected_dashboard)


def test__get_view_three_part_key(three_filter_dashboard: DashboardFixture):
    _, dashboard = three_filter_dashboard
    assert dashboard._get_view(
        (
            ("TemporalGrouper_%B", "=", "January"),
            ("TemporalGrouper_%Y", "=", "2022"),
            ("subject_name", "=", "jo"),
        )
    ) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/jan/2022/jo/map.html",
            is_filtered=True,
        ),
    ]
    assert dashboard._get_view(
        (
            ("TemporalGrouper_%B", "=", "January"),
            ("TemporalGrouper_%Y", "=", "2022"),
            ("subject_name", "=", "zo"),
        )
    ) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/jan/2022/zo/map.html",
            is_filtered=True,
        ),
    ]


def test_model_dump_views_three_filter(three_filter_dashboard: DashboardFixture):
    _, dashboard = three_filter_dashboard
    assert dashboard.model_dump()["views"] == {
        # Note sort_keys=True in `json.dumps` in _iter_views_json method of Dashboard
        # results in the json keys being ordered differently than the keys in the tuple
        '{"TemporalGrouper_%B": "January", "TemporalGrouper_%Y": "2022", "subject_name": "jo"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/jan/2022/jo/map.html",
                "is_filtered": True,
            },
        ],
        '{"TemporalGrouper_%B": "January", "TemporalGrouper_%Y": "2022", "subject_name": "zo"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/jan/2022/zo/map.html",
                "is_filtered": True,
            },
        ],
    }


def test_model_dump_filters_three_filter(three_filter_dashboard: DashboardFixture):
    _, dashboard = three_filter_dashboard
    assert dashboard.model_dump()["filters"] == {
        "schema": {
            "type": "object",
            "properties": {
                "TemporalGrouper_%B": {
                    "type": "string",
                    "title": "Month",
                    "oneOf": [
                        {"const": "January", "title": "January"},
                    ],
                    "default": "January",
                },
                "TemporalGrouper_%Y": {
                    "type": "string",
                    "title": "Year",
                    "oneOf": [
                        {"const": "2022", "title": "2022"},
                    ],
                    "default": "2022",
                },
                "subject_name": {
                    "type": "string",
                    "title": "Subject Name",
                    "oneOf": [
                        {"const": "jo", "title": "jo"},
                        {"const": "zo", "title": "zo"},
                    ],
                    "default": "jo",
                },
            },
            "uiSchema": {
                "TemporalGrouper_%B": {
                    "ui:title": "Month",
                    "ui:widget": "select",
                },
                "TemporalGrouper_%Y": {
                    "ui:title": "Year",
                    "ui:widget": "select",
                },
                "subject_name": {
                    "ui:title": "Subject Name",
                    "ui:widget": "select",
                },
            },
        }
    }


@pytest.fixture
def dashboard_with_none_views() -> DashboardFixture:
    great_map = GroupedWidget(
        widget_type="map",
        title="A Great Map",
        views={
            (("TemporalGrouper_%B", "=", "January"),): "/path/to/precomputed/jan/map.html",
            (("TemporalGrouper_%B", "=", "February"),): "/path/to/precomputed/feb/map.html",
        },
        is_filtered=True,
    )
    none_view_plot = GroupedWidget(
        widget_type="graph",
        title="A plot with only one view and no groupers",
        views={
            None: "/path/to/precomputed/single/plot.html",
        },
        is_filtered=False,
    )
    widgets = [great_map, none_view_plot]
    dashboard = Dashboard(
        grouper_choices={TemporalGrouper(temporal_index=Month()): ["January", "February"]},
        keys=[
            (("TemporalGrouper_%B", "=", "February"),),
            (("TemporalGrouper_%B", "=", "January"),),
        ],
        widgets=widgets,
        metadata=Metadata(
            title="A Great Dashboard",
            description="A dashboard with a map and a plot",
            time_range="From 01 Jan 2011 00:00:00 to 01 Jan 2023 00:00:00",
            time_zone="UTC",
        ),
    )
    return widgets, dashboard


def test_gather_dashboard_with_none_views(dashboard_with_none_views: DashboardFixture):
    grouped_widgets, expected_dashboard = dashboard_with_none_views
    dashboard: Dashboard = gather_dashboard(
        details=WorkflowDetails(
            name="A Great Dashboard",
            description="A dashboard with a map and a plot",
        ),
        time_range=TimeRange(
            since=datetime.strptime("2011-01-01", "%Y-%m-%d"),
            until=datetime.strptime("2023-01-01", "%Y-%m-%d"),
            timezone=UTC_TIMEZONEINFO,
        ),
        widgets=grouped_widgets,
        groupers=[TemporalGrouper(temporal_index=Month())],
    )
    assert_dashboards_equal(dashboard, expected_dashboard)


def test__get_view_with_none_views(dashboard_with_none_views: DashboardFixture):
    _, dashboard = dashboard_with_none_views
    assert dashboard._get_view((("TemporalGrouper_%B", "=", "January"),)) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/precomputed/jan/map.html",
            is_filtered=True,
        ),
        EmumeratedWidgetSingleView(
            id=1,
            widget_type="graph",
            title="A plot with only one view and no groupers",
            data="/path/to/precomputed/single/plot.html",
            is_filtered=False,
        ),
    ]
    assert dashboard._get_view((("TemporalGrouper_%B", "=", "February"),)) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A Great Map",
            data="/path/to/precomputed/feb/map.html",
            is_filtered=True,
        ),
        EmumeratedWidgetSingleView(
            id=1,
            widget_type="graph",
            title="A plot with only one view and no groupers",
            data="/path/to/precomputed/single/plot.html",
            is_filtered=False,
        ),
    ]


def test_model_dump_views_with_none_views(dashboard_with_none_views: DashboardFixture):
    _, dashboard = dashboard_with_none_views
    assert dashboard.model_dump()["views"] == {
        '{"TemporalGrouper_%B": "January"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/precomputed/jan/map.html",
                "is_filtered": True,
            },
            {
                "id": 1,
                "widget_type": "graph",
                "title": "A plot with only one view and no groupers",
                "data": "/path/to/precomputed/single/plot.html",
                "is_filtered": False,
            },
        ],
        '{"TemporalGrouper_%B": "February"}': [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A Great Map",
                "data": "/path/to/precomputed/feb/map.html",
                "is_filtered": True,
            },
            {
                "id": 1,
                "widget_type": "graph",
                "title": "A plot with only one view and no groupers",
                "data": "/path/to/precomputed/single/plot.html",
                "is_filtered": False,
            },
        ],
    }


def test_model_dump_filters_with_none_views(
    dashboard_with_none_views: DashboardFixture,
):
    _, dashboard = dashboard_with_none_views
    assert dashboard.model_dump()["filters"] == {
        "schema": {
            "type": "object",
            "properties": {
                "TemporalGrouper_%B": {
                    "type": "string",
                    "title": "Month",
                    "oneOf": [
                        {"const": "January", "title": "January"},
                        {"const": "February", "title": "February"},
                    ],
                    "default": "January",
                },
            },
            "uiSchema": {
                "TemporalGrouper_%B": {
                    "ui:title": "Month",
                    "ui:widget": "select",
                },
            },
        }
    }


@pytest.fixture
def dashboard_with_all_none_views() -> DashboardFixture:
    none_view_map = GroupedWidget(
        widget_type="map",
        title="A map with only one view and no groupers",
        views={
            None: "/path/to/precomputed/single/map.html",
        },
        is_filtered=False,
    )
    none_view_plot = GroupedWidget(
        widget_type="graph",
        title="A plot with only one view and no groupers",
        views={
            None: "/path/to/precomputed/single/plot.html",
        },
        is_filtered=False,
    )
    widgets = [none_view_map, none_view_plot]
    dashboard = Dashboard(widgets=widgets)
    return widgets, dashboard


def test_gather_dashboard_with_all_none_views(
    dashboard_with_all_none_views: DashboardFixture,
):
    grouped_widgets, expected_dashboard = dashboard_with_all_none_views
    dashboard: Dashboard = gather_dashboard(
        details=WorkflowDetails(
            name=expected_dashboard.metadata.title,
            description=expected_dashboard.metadata.description,
        ),
        time_range=None,
        widgets=grouped_widgets,
        groupers=None,
    )
    # We don't need to use the custom `assert_dashboards_equal` function here
    # because there are no groupers or keys (with sorting concerns) to compare
    assert dashboard == expected_dashboard


def test__get_view_with_all_none_views(dashboard_with_all_none_views: DashboardFixture):
    _, dashboard = dashboard_with_all_none_views
    assert dashboard._get_view(None) == [
        EmumeratedWidgetSingleView(
            id=0,
            widget_type="map",
            title="A map with only one view and no groupers",
            data="/path/to/precomputed/single/map.html",
            is_filtered=False,
        ),
        EmumeratedWidgetSingleView(
            id=1,
            widget_type="graph",
            title="A plot with only one view and no groupers",
            data="/path/to/precomputed/single/plot.html",
            is_filtered=False,
        ),
    ]


def test_model_dump_views_with_all_none_views(
    dashboard_with_all_none_views: DashboardFixture,
):
    _, dashboard = dashboard_with_all_none_views
    assert dashboard.model_dump()["views"] == {
        "{}": [
            {
                "id": 0,
                "widget_type": "map",
                "title": "A map with only one view and no groupers",
                "data": "/path/to/precomputed/single/map.html",
                "is_filtered": False,
            },
            {
                "id": 1,
                "widget_type": "graph",
                "title": "A plot with only one view and no groupers",
                "data": "/path/to/precomputed/single/plot.html",
                "is_filtered": False,
            },
        ],
    }


def test_model_dump_filters_with_all_none_views(
    dashboard_with_all_none_views: DashboardFixture,
):
    _, dashboard = dashboard_with_all_none_views
    assert dashboard.model_dump()["filters"] is None


@pytest.fixture
def outer_join_dashboard() -> Dashboard:
    jan_map = GroupedWidget(
        widget_type="map",
        title="A great map that happens to have only january data",
        views={(("TemporalGrouper_%B", "=", "January"),): "/path/to/precomputed/jan/map.html"},
        is_filtered=True,
    )
    feb_plot = GroupedWidget(
        widget_type="graph",
        title="A plot that happens to have only february data",
        views={
            (("TemporalGrouper_%B", "=", "February"),): "/path/to/precomputed/feb/plot.html",
        },
        is_filtered=False,
    )
    dashboard = gather_dashboard(
        details=WorkflowDetails(name="Demonstration of outer join behavior", description=""),
        time_range=TimeRange(
            since=datetime.strptime("2011-01-01", "%Y-%m-%d"),
            until=datetime.strptime("2011-03-01", "%Y-%m-%d"),
            timezone=UTC_TIMEZONEINFO,
        ),
        widgets=[jan_map, feb_plot],
        groupers=[TemporalGrouper(temporal_index=Month())],
    )
    return dashboard


def test_gather_dashboard_views_outer_join(outer_join_dashboard: Dashboard):
    dashboard_json = outer_join_dashboard.model_dump()
    assert dashboard_json["views"]['{"TemporalGrouper_%B": "January"}'] == [
        {
            "id": 0,
            "widget_type": "map",
            "title": "A great map that happens to have only january data",
            "data": "/path/to/precomputed/jan/map.html",
            "is_filtered": True,
        },
        {
            "id": 1,
            "widget_type": "graph",
            "title": "A plot that happens to have only february data",
            "data": None,  # this is the outer join behavior in action
            "is_filtered": False,
        },
    ]
    assert dashboard_json["views"]['{"TemporalGrouper_%B": "February"}'] == [
        {
            "id": 0,
            "widget_type": "map",
            "title": "A great map that happens to have only january data",
            "data": None,  # and this is the outer join behavior in action
            "is_filtered": True,
        },
        {
            "id": 1,
            "widget_type": "graph",
            "title": "A plot that happens to have only february data",
            "data": "/path/to/precomputed/feb/plot.html",
            "is_filtered": False,
        },
    ]
    # and note that the filters include all the months, even though no individual
    # widget has data for all of the months. this is also the outer join behavior.
    assert dashboard_json["filters"]["schema"]["properties"]["TemporalGrouper_%B"] == {
        "type": "string",
        "title": "Month",
        "oneOf": [
            {"const": "January", "title": "January"},
            {"const": "February", "title": "February"},
        ],
        "default": "January",
    }


def test_gather_dashboard_with_warning(single_filter_dashboard: DashboardFixture):
    grouped_widgets, expected_dashboard = single_filter_dashboard
    expected_dashboard.metadata.warning = "This dashboard might be too great"

    dashboard: Dashboard = gather_dashboard(
        details=WorkflowDetails(
            name="A Great Dashboard",
            description="A dashboard with a map and a plot",
        ),
        time_range=TimeRange(
            since=datetime.strptime("2011-01-01", "%Y-%m-%d"),
            until=datetime.strptime("2023-01-01", "%Y-%m-%d"),
            timezone=UTC_TIMEZONEINFO,
        ),
        widgets=grouped_widgets,
        groupers=[TemporalGrouper(temporal_index=Month())],
        warning="This dashboard might be too great",
    )
    assert_dashboards_equal(dashboard, expected_dashboard)


def test_metadata_serialization_excludes_empty_fields():
    full_metadata = Metadata(
        title="Test",
        description="Test description",
        time_range="From 01 Jan 2011 to 01 Jan 2023",
        time_zone="UTC",
        warning="Danger!",
    )
    model_dump_full = full_metadata.model_dump()
    assert "time_range" in model_dump_full
    assert "time_zone" in model_dump_full
    assert model_dump_full["time_range"] == "From 01 Jan 2011 to 01 Jan 2023"
    assert model_dump_full["time_zone"] == "UTC"

    partial_metadata = Metadata(
        title="Test",
        description="Test description",
    )
    model_dump_partial = partial_metadata.model_dump()
    assert "time_range" not in model_dump_partial
    assert "time_zone" not in model_dump_partial
    assert model_dump_partial == {"title": "Test", "description": "Test description"}


def test_gather_dashboard_with_no_time_range(single_filter_dashboard: DashboardFixture):
    grouped_widgets, expected_dashboard = single_filter_dashboard

    expected_dashboard.metadata.time_range = ""
    expected_dashboard.metadata.time_zone = ""

    dashboard: Dashboard = gather_dashboard(
        details=WorkflowDetails(
            name="A Great Dashboard",
            description="A dashboard with a map and a plot",
        ),
        widgets=grouped_widgets,
        groupers=[TemporalGrouper(temporal_index=Month())],
    )
    assert_dashboards_equal(dashboard, expected_dashboard)
