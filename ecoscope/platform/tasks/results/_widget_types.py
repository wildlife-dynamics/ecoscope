from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel
from pydantic_core import Url

from ecoscope.platform.indexes import CompositeFilter
from ecoscope.platform.tasks.results._ecoplot import PlotlyJsonSpec
from ecoscope.platform.tasks.results._pydeck import DeckJsonSpec

WidgetTypes = Literal["graph", "map", "text", "stat", "table", "map_v2", "graph_v2", "table_v2"]


class TableV2WidgetData(BaseModel):
    """A persisted table for native rendering, plus the `TableConfig` display options."""

    url: str
    enable_sorting: bool = True
    enable_filtering: bool = False
    enable_download: bool = False
    hide_header: bool = False


PrecomputedHTMLWidgetData: TypeAlias = Path | Url | None
TextWidgetData: TypeAlias = str | None
MapWidgetData: TypeAlias = DeckJsonSpec | None
SingleValueWidgetData: TypeAlias = str | None
PlotV2WidgetData: TypeAlias = PlotlyJsonSpec | None
TableV2WidgetDataOrNone: TypeAlias = TableV2WidgetData | None

WidgetData: TypeAlias = (
    PrecomputedHTMLWidgetData
    | TextWidgetData
    | MapWidgetData
    | SingleValueWidgetData
    | PlotV2WidgetData
    | TableV2WidgetDataOrNone
)
GroupedWidgetMergeKey: TypeAlias = tuple[str, str]


@dataclass
class WidgetBase:
    widget_type: WidgetTypes
    title: str
    is_filtered: bool


@dataclass
class WidgetSingleView(WidgetBase):
    data: WidgetData
    view: CompositeFilter | None = None


@dataclass
class GroupedWidget(WidgetBase):
    views: dict[CompositeFilter | None, WidgetData]

    @classmethod
    def from_single_view(cls, view: WidgetSingleView) -> "GroupedWidget":
        """Construct a GroupedWidget from a WidgetSingleView.
        The resulting GroupedWidget will have a single view.
        """
        return cls(
            widget_type=view.widget_type,
            title=view.title,
            views={view.view: view.data},
            is_filtered=view.is_filtered,
        )

    def add_null_view(self, view_key: CompositeFilter) -> None:
        """Add a null view to the GroupedWidget for the given key."""
        assert self.views.get(view_key) is None, f"View already exists for {view_key=}"
        self.views[view_key] = None

    def get_view(self, view: CompositeFilter | None) -> WidgetSingleView:
        """Get a WidgetSingleView for a specific view."""
        if view not in self.views:
            raise ValueError(f"Requested {view=} not found in {self.views=}")
        return WidgetSingleView(
            widget_type=self.widget_type,
            title=self.title,
            view=view,
            data=self.views[view],
            is_filtered=self.is_filtered,
        )

    @property
    def merge_key(self) -> GroupedWidgetMergeKey:
        """If two GroupedWidgets have the same merge key, they can be merged."""
        return (self.widget_type, self.title)

    def __ior__(self, other: "GroupedWidget") -> "GroupedWidget":
        """Implements the in-place or operator, i.e. `|=`, used to merge two GroupedWidgets."""
        if self.merge_key != other.merge_key:
            raise ValueError(
                f"Cannot merge GroupedWidgets with different merge keys: {self.merge_key} != {other.merge_key}"
            )
        self.views.update(other.views)
        return self


@dataclass
class FilesListSingleView:
    """A list of files associated with a single view of a dashboard.
    Mirrors `WidgetSingleView`, but carries a list of file paths rather than widget data.
    """

    files: list[Path]
    view: CompositeFilter | None = None


@dataclass
class GroupedFiles:
    """The files associated with a dashboard across all of its views. Mirrors `GroupedWidget`,
    but the per-view value is a list of file paths. Unlike a widget, files are not titled, so
    a dashboard has a single `GroupedFiles` and there is no `merge_key`.
    """

    views: dict[CompositeFilter | None, list[Path]]

    @classmethod
    def from_single_view(cls, view: FilesListSingleView) -> "GroupedFiles":
        """Construct a GroupedFiles from a FilesListSingleView.
        The resulting GroupedFiles will have a single view.
        """
        return cls(views={view.view: view.files})

    def get_view(self, view: CompositeFilter | None) -> list[Path]:
        """Get the list of files for a specific view, or an empty list if there are none."""
        return self.views.get(view, [])

    def __ior__(self, other: "GroupedFiles") -> "GroupedFiles":
        """Implements the in-place or operator, i.e. `|=`, used to merge two GroupedFiles.
        Files for the same view key are concatenated rather than overwritten.
        """
        for key, paths in other.views.items():
            self.views.setdefault(key, []).extend(paths)
        return self
