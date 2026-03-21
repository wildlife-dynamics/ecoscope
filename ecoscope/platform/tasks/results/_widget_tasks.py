from typing import Annotated

from pydantic import Field
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import AdvancedField
from ecoscope.platform.indexes import CompositeFilter
from ecoscope.platform.tasks.results._widget_types import (
    GroupedWidget,
    GroupedWidgetMergeKey,
    PrecomputedHTMLWidgetData,
    TextWidgetData,
    WidgetData,
    WidgetSingleView,
)
from ecoscope.platform.tasks.transformation._unit import Quantity


def _fallback_to_none(obj: WidgetData | SkipSentinel) -> WidgetData | None:
    """Fallback function to convert SkipSentinel to None."""
    return None if isinstance(obj, SkipSentinel) else obj


@register()
def create_map_widget_single_view(
    title: Annotated[str, Field(description="The title of the widget")],
    data: Annotated[
        PrecomputedHTMLWidgetData,
        Field(description="Path to precomputed HTML"),
        SkippedDependencyFallback(_fallback_to_none),
    ],
    view: Annotated[
        CompositeFilter | None,
        Field(description="If grouped, the view of the widget", exclude=True),
    ] = None,
) -> Annotated[WidgetSingleView, Field(description="The widget")]:
    """Create a map widget with a single view.

    Args:
        title: The title of the widget.
        data: Path to precomputed HTML.
        view: If grouped, the view of the widget.

    Returns:
        The widget.
    """
    return WidgetSingleView(
        widget_type="map",
        title=title,
        view=view,
        data=data,
        is_filtered=(view is not None),
    )


@register()
def create_plot_widget_single_view(
    title: Annotated[str, Field(description="The title of the widget")],
    data: Annotated[
        PrecomputedHTMLWidgetData,
        Field(description="Path to precomputed HTML"),
        SkippedDependencyFallback(_fallback_to_none),
    ],
    view: Annotated[
        CompositeFilter | None,
        Field(description="If grouped, the view of the widget", exclude=True),
    ] = None,
) -> Annotated[WidgetSingleView, Field(description="The widget")]:
    """Create a plot widget with a single view.

    Args:
        title: The title of the widget.
        data: Path to precomputed HTML.
        view: If grouped, the view of the widget.

    Returns:
        The widget.
    """
    return WidgetSingleView(
        widget_type="graph",
        title=title,
        view=view,
        data=data,
        is_filtered=(view is not None),
    )


@register()
def create_text_widget_single_view(
    title: Annotated[str, Field(description="The title of the widget")],
    data: Annotated[
        TextWidgetData,
        Field(description="Text to display."),
        SkippedDependencyFallback(_fallback_to_none),
    ],
    view: Annotated[
        CompositeFilter | None,
        Field(description="If grouped, the view of the widget", exclude=True),
    ] = None,
) -> Annotated[WidgetSingleView, Field(description="The widget")]:
    """Create a text widget with a single view.

    Args:
        title: The title of the widget.
        data: Text to display.
        view: If grouped, the view of the widget.

    Returns:
        The widget.
    """
    return WidgetSingleView(
        widget_type="text",
        title=title,
        view=view,
        data=data,
        is_filtered=(view is not None),
    )


@register()
def create_single_value_widget_single_view(
    title: Annotated[str, Field(description="The title of the widget")],
    data: Annotated[
        Quantity | float | int | None,
        Field(description="Value to display."),
        SkippedDependencyFallback(_fallback_to_none),
    ],
    view: Annotated[
        CompositeFilter | None,
        Field(description="If grouped, the view of the widget", exclude=True),
    ] = None,
    decimal_places: Annotated[
        int,
        AdvancedField(default=1, description="The number of decimal places to display."),
    ] = 1,
) -> Annotated[WidgetSingleView, Field(description="The widget")]:
    """Create a single value widget with a single view.

    Args:
        title: The title of the widget.
        data: The value to display.
        view: If grouped, the view of the widget.
        decimal_places: The number of decimal places to display.

    Returns:
        The widget.
    """
    data_str = ""
    if data is not None:
        if isinstance(data, Quantity):
            data_str = f"{data.value:.{decimal_places}f} {data.unit or ''}".strip()
        elif isinstance(data, float):
            data_str = f"{data:.{decimal_places}f}"
        else:
            data_str = str(data)

    return WidgetSingleView(
        widget_type="stat",
        title=title,
        view=view,
        data=(data_str if data is not None else None),
        is_filtered=(view is not None),
    )


@register()
def create_table_widget_single_view(
    title: Annotated[str, Field(description="The title of the widget")],
    data: Annotated[
        PrecomputedHTMLWidgetData,
        Field(description="Path to precomputed HTML"),
        SkippedDependencyFallback(_fallback_to_none),
    ],
    view: Annotated[
        CompositeFilter | None,
        Field(description="If grouped, the view of the widget", exclude=True),
    ] = None,
) -> Annotated[WidgetSingleView, Field(description="The widget")]:
    """Create a table widget with a single view.

    Args:
        title: The title of the widget.
        data: Path to precomputed HTML.
        view: If grouped, the view of the widget.

    Returns:
        The widget.
    """
    return WidgetSingleView(
        widget_type="table",
        title=title,
        view=view,
        data=data,
        is_filtered=(view is not None),
    )


@register()
def create_mapv2_widget_single_view(
    title: Annotated[str, Field(description="The title of the widget")],
    data: Annotated[
        PrecomputedHTMLWidgetData,
        Field(description="Path to precomputed HTML"),
        SkippedDependencyFallback(_fallback_to_none),
    ],
    view: Annotated[
        CompositeFilter | None,
        Field(description="If grouped, the view of the widget", exclude=True),
    ] = None,
) -> Annotated[WidgetSingleView, Field(description="The widget")]:
    """Create a mapV2 widget with a single view.

    Args:
        title: The title of the widget.
        data: Path to precomputed HTML.
        view: If grouped, the view of the widget.

    Returns:
        The widget.
    """
    return WidgetSingleView(
        widget_type="mapV2",
        title=title,
        view=view,
        data=data,
        is_filtered=(view is not None),
    )


@register()
def merge_widget_views(
    widgets: Annotated[
        list[WidgetSingleView],
        Field(description="The widgets to merge", exclude=True),
    ],
) -> Annotated[list[GroupedWidget], Field(description="The merged widgets")]:
    """Merge widgets with the same `title` and `widget_type`.

    Args:
        widgets: The widgets to merge.

    Returns:
        The merged grouped widgets.
    """
    grouped_widgets = [GroupedWidget.from_single_view(w) for w in widgets]
    merged: dict[GroupedWidgetMergeKey, GroupedWidget] = {}
    for gw in grouped_widgets:
        if gw.merge_key not in merged:
            merged[gw.merge_key] = gw
        else:
            merged[gw.merge_key] |= gw
    return list(merged.values())
