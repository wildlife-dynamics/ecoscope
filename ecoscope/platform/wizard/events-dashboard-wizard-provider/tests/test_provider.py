"""Tests for EcoscopeEventsDashboardWizardProvider and associated validators."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from ecoscope_events_dashboard_wizard_provider import EcoscopeEventsDashboardWizardProvider
from ecoscope_events_dashboard_wizard_provider.provider import (
    WIDGET_CHOICES,
    _widget_batch_type,
    widget_title_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def drive_wizard(
    provider: EcoscopeEventsDashboardWizardProvider,
    answers: list[str | None],
) -> list[dict]:
    """Drive wizard generator with a sequence of answers.

    Returns all yielded questions (mirrors wt-compiler conftest helper).
    """
    gen = provider.input_generator()
    questions: list[dict] = []
    try:
        q = next(gen)
        questions.append(q)
        for answer in answers:
            q = gen.send(answer)
            questions.append(q)
    except StopIteration:
        pass
    return questions


def make_provider_with_widgets(
    widgets: list[dict[str, Any]],
) -> EcoscopeEventsDashboardWizardProvider:
    """Drive a provider through default questions and the widgets loop.

    Args:
        widgets: List of dicts with ``widget`` and ``title`` keys.

    Returns:
        A provider whose answers include all default fields and ``widgets``.
    """
    provider = EcoscopeEventsDashboardWizardProvider()
    # Build answers for default questions: workflow_id, workflow_name,
    # workflow_description, author_name, license_type, then empty requirements loop.
    base_answers: list[str | None] = [
        "my_dashboard",  # workflow_id
        "My Dashboard",  # workflow_name
        "Dashboard description",  # workflow_description
        "Test Author",  # author_name
        "MIT",  # license_type
        None,  # terminate requirements loop
    ]
    # Now drive the widgets loop: for each widget, send widget type then title,
    # then terminate with None for the sentinel (widget type).
    widget_answers: list[str | None] = []
    for w in widgets:
        widget_answers.append(w["widget"])  # sentinel question answer
        widget_answers.append(w["title"])  # title question answer
    widget_answers.append(None)  # terminate loop
    drive_wizard(provider, base_answers + widget_answers)
    return provider


def render_templates(
    widgets: list[dict[str, Any]],
) -> dict[str, str]:
    """Render spec.yaml and layout.json for the given widgets, return file contents.

    Args:
        widgets: List of dicts with ``widget`` and ``title`` keys.

    Returns:
        Dict with keys ``spec`` and ``layout`` containing rendered file text.
    """
    provider = make_provider_with_widgets(widgets)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        provider.dump(out)
        return {
            "spec": (out / "spec.yaml").read_text(),
            "layout": (out / "layout.json").read_text(),
        }


# ---------------------------------------------------------------------------
# widget_title_type
# ---------------------------------------------------------------------------


class TestWidgetTitleType:
    """Tests for widget_title_type validator."""

    def test_valid_title_returned(self) -> None:
        assert widget_title_type("Events Map") == "Events Map"

    def test_strips_whitespace(self) -> None:
        assert widget_title_type("  Events Map  ") == "Events Map"

    def test_empty_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="empty"):
            widget_title_type("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="empty"):
            widget_title_type("   ")

    def test_exactly_50_chars_valid(self) -> None:
        title = "A" * 50
        assert widget_title_type(title) == title

    def test_51_chars_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="50 characters"):
            widget_title_type("A" * 51)

    def test_unicode_title_valid(self) -> None:
        title = "Karte der Ereignisse"
        assert widget_title_type(title) == title


# ---------------------------------------------------------------------------
# _widget_batch_type
# ---------------------------------------------------------------------------


class TestWidgetBatchType:
    """Tests for _widget_batch_type JSON validator."""

    def test_valid_object_parsed(self) -> None:
        d = _widget_batch_type('{"widget": "bar_chart", "title": "My Bar Chart"}')
        assert d["widget"] == "bar_chart"
        assert d["title"] == "My Bar Chart"

    def test_all_widget_choices_accepted(self) -> None:
        for choice in WIDGET_CHOICES:
            d = _widget_batch_type(f'{{"widget": "{choice}", "title": "Title"}}')
            assert d["widget"] == choice

    def test_invalid_widget_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="unknown"):
            _widget_batch_type('{"widget": "unknown", "title": "X"}')

    def test_missing_widget_field_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="widget"):
            _widget_batch_type('{"title": "My Title"}')

    def test_missing_title_field_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="title"):
            _widget_batch_type('{"widget": "bar_chart"}')

    def test_title_too_long_raises(self) -> None:
        payload = json.dumps({"widget": "bar_chart", "title": "A" * 51})
        with pytest.raises(argparse.ArgumentTypeError, match="50 characters"):
            _widget_batch_type(payload)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="JSON"):
            _widget_batch_type("not-json")

    def test_json_array_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="object"):
            _widget_batch_type('[{"widget": "bar_chart", "title": "X"}]')

    def test_title_stripped_whitespace(self) -> None:
        d = _widget_batch_type('{"widget": "events_map", "title": "  My Map  "}')
        assert d["title"] == "My Map"


# ---------------------------------------------------------------------------
# Question structure
# ---------------------------------------------------------------------------


class TestQuestionStructure:
    """Tests for provider question list structure."""

    def test_last_question_is_widgets_loop(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        last = questions[-1]
        assert last["dest"] == "widgets"

    def test_widgets_loop_has_widget_sub_question(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        loop = questions[-1]
        assert loop["questions"][0]["dest"] == "widget"

    def test_widgets_loop_sentinel_choices(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        loop = questions[-1]
        sentinel = loop["questions"][0]
        assert sentinel["argparse"]["choices"] == WIDGET_CHOICES

    def test_widgets_loop_has_title_sub_question(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        loop = questions[-1]
        assert loop["questions"][1]["dest"] == "title"

    def test_inherits_default_questions(self) -> None:
        from wt_compiler.wizard.default import DefaultWizardProvider

        default = DefaultWizardProvider()
        provider = EcoscopeEventsDashboardWizardProvider()
        default_dests = [q["dest"] for q in default.get_questions()]
        provider_dests = [q["dest"] for q in provider.get_questions()]
        for dest in default_dests:
            assert dest in provider_dests

    def test_get_questions_returns_independent_copies(self) -> None:
        """Each call to get_questions() returns a fresh deep copy of the loop."""
        p = EcoscopeEventsDashboardWizardProvider()
        q1 = p.get_questions()
        q2 = p.get_questions()
        assert q1[-1] is not q2[-1]


# ---------------------------------------------------------------------------
# dump() validation
# ---------------------------------------------------------------------------


class TestDumpValidation:
    """Tests for dump() widget validation."""

    def test_dump_raises_on_empty_widgets(self, tmp_path: Path) -> None:
        """dump() raises ValueError when no widgets are selected."""
        provider = EcoscopeEventsDashboardWizardProvider()
        base_answers: list[str | None] = [
            "my_dashboard",
            "My Dashboard",
            "Dashboard description",
            "Test Author",
            "MIT",
            None,  # terminate requirements loop
            None,  # terminate widgets loop immediately
        ]
        drive_wizard(provider, base_answers)
        with pytest.raises(ValueError, match="[Aa]t least one widget"):
            provider.dump(tmp_path)

    def test_dump_raises_on_duplicate_widget_type(self, tmp_path: Path) -> None:
        """dump() raises ValueError when a duplicate widget type is present."""
        provider = EcoscopeEventsDashboardWizardProvider()
        base_answers: list[str | None] = [
            "my_dashboard",
            "My Dashboard",
            "Dashboard description",
            "Test Author",
            "MIT",
            None,  # terminate requirements loop
            "bar_chart",
            "First Bar Chart",
            "bar_chart",
            "Second Bar Chart",
            None,  # terminate widgets loop
        ]
        drive_wizard(provider, base_answers)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            provider.dump(tmp_path)

    def test_dump_succeeds_with_one_widget(self, tmp_path: Path) -> None:
        """dump() succeeds and writes files when exactly one widget is selected."""
        provider = make_provider_with_widgets([{"widget": "bar_chart", "title": "Events"}])
        provider.dump(tmp_path)
        assert (tmp_path / "spec.yaml").exists()
        assert (tmp_path / "layout.json").exists()


# ---------------------------------------------------------------------------
# Template rendering — spec.yaml
# ---------------------------------------------------------------------------


class TestSpecRendering:
    """Tests for spec.yaml template rendering."""

    def test_bar_chart_tasks_present_when_enabled(self) -> None:
        result = render_templates([{"widget": "bar_chart", "title": "My Bar Chart"}])
        assert "events_bar_chart" in result["spec"]
        assert "grouped_bar_plot_widget_merge" in result["spec"]

    def test_events_map_tasks_present_when_enabled(self) -> None:
        result = render_templates([{"widget": "events_map", "title": "My Map"}])
        assert "grouped_events_map_layer" in result["spec"]
        assert "grouped_events_map_widget_merge" in result["spec"]

    def test_pie_chart_tasks_present_when_enabled(self) -> None:
        result = render_templates([{"widget": "pie_chart", "title": "My Pie"}])
        assert "grouped_events_pie_chart" in result["spec"]
        assert "grouped_events_pie_widget_merge" in result["spec"]

    def test_event_count_map_tasks_present_when_enabled(self) -> None:
        result = render_templates([{"widget": "event_count_map", "title": "Heatmap"}])
        assert "events_meshgrid" in result["spec"]
        assert "grouped_fd_map_widget_merge" in result["spec"]

    def test_events_table_tasks_present_when_enabled(self) -> None:
        result = render_templates([{"widget": "events_table", "title": "Event List"}])
        assert "events_table" in result["spec"]
        assert "grouped_table_widget_merge" in result["spec"]

    def test_disabled_widget_tasks_absent(self) -> None:
        """Only bar_chart selected — event_count_map tasks must not appear."""
        result = render_templates([{"widget": "bar_chart", "title": "Chart"}])
        assert "events_meshgrid" not in result["spec"]
        assert "grouped_events_pie_chart" not in result["spec"]
        assert "grouped_events_map_layer" not in result["spec"]
        assert "grouped_table_widget_merge" not in result["spec"]

    def test_custom_title_appears_in_set_string_var(self) -> None:
        result = render_templates([{"widget": "bar_chart", "title": "My Custom Title"}])
        assert "My Custom Title" in result["spec"]

    def test_gather_dashboard_widgets_in_loop_order(self) -> None:
        """gather_dashboard.widgets respects user-defined order, not canonical order."""
        result = render_templates(
            [
                {"widget": "events_map", "title": "Events Map"},
                {"widget": "bar_chart", "title": "Bar Chart"},
            ]
        )
        spec = result["spec"]
        map_pos = spec.index("grouped_events_map_widget_merge")
        bar_pos = spec.index("grouped_bar_plot_widget_merge")
        # In gather_dashboard section, map merge should appear before bar merge
        gather_pos = spec.index("gather_dashboard")
        # Find occurrences after gather_dashboard
        map_in_gather = spec.index("grouped_events_map_widget_merge", gather_pos)
        bar_in_gather = spec.index("grouped_bar_plot_widget_merge", gather_pos)
        assert map_in_gather < bar_in_gather

    def test_all_five_widgets_render_all_tasks(self) -> None:
        widgets = [
            {"widget": "events_map", "title": "Events Map"},
            {"widget": "bar_chart", "title": "Bar Chart"},
            {"widget": "pie_chart", "title": "Pie Chart"},
            {"widget": "event_count_map", "title": "Heatmap"},
            {"widget": "events_table", "title": "Table"},
        ]
        result = render_templates(widgets)
        spec = result["spec"]
        assert "grouped_bar_plot_widget_merge" in spec
        assert "grouped_events_map_widget_merge" in spec
        assert "grouped_events_pie_widget_merge" in spec
        assert "grouped_fd_map_widget_merge" in spec
        assert "grouped_table_widget_merge" in spec

    def test_rename_display_columns_present_for_map(self) -> None:
        result = render_templates([{"widget": "events_map", "title": "Map"}])
        assert "rename_display_columns" in result["spec"]

    def test_rename_display_columns_present_for_table(self) -> None:
        result = render_templates([{"widget": "events_table", "title": "Table"}])
        assert "rename_display_columns" in result["spec"]

    def test_rename_display_columns_absent_for_bar_only(self) -> None:
        result = render_templates([{"widget": "bar_chart", "title": "Chart"}])
        assert "rename_display_columns" not in result["spec"]

    def test_base_map_defs_task_present_for_map_widget(self) -> None:
        result = render_templates([{"widget": "events_map", "title": "Map"}])
        assert "set_base_maps" in result["spec"]

    def test_base_map_defs_task_present_for_event_count_map(self) -> None:
        result = render_templates([{"widget": "event_count_map", "title": "Heatmap"}])
        assert "set_base_maps" in result["spec"]

    def test_base_map_defs_task_absent_when_no_map_widgets(self) -> None:
        """When no map widgets are selected, the set_base_maps task is not rendered."""
        result = render_templates(
            [
                {"widget": "bar_chart", "title": "Chart"},
                {"widget": "pie_chart", "title": "Pie"},
            ]
        )
        # The task references set_base_maps — absent when no map widgets enabled
        assert "set_base_maps" not in result["spec"]

    def test_wt_compiler_references_not_evaluated(self) -> None:
        """${{ }} references must appear literally in rendered output."""
        result = render_templates([{"widget": "bar_chart", "title": "Chart"}])
        assert "${{ workflow." in result["spec"]

    def test_workflow_id_in_spec(self) -> None:
        result = render_templates([{"widget": "bar_chart", "title": "Chart"}])
        assert "my_dashboard" in result["spec"]

    def test_fd_uischema_absent_when_event_count_map_disabled(self) -> None:
        result = render_templates([{"widget": "bar_chart", "title": "Chart"}])
        assert "auto_scale_or_custom_cell_size" not in result["spec"]

    def test_fd_uischema_present_when_event_count_map_enabled(self) -> None:
        result = render_templates([{"widget": "event_count_map", "title": "Heatmap"}])
        assert "auto_scale_or_custom_cell_size" in result["spec"]


# ---------------------------------------------------------------------------
# Template rendering — layout.json
# ---------------------------------------------------------------------------


class TestLayoutRendering:
    """Tests for layout.json template rendering."""

    def _parse_layout(self, layout_text: str) -> list[dict]:
        return json.loads(layout_text)

    def test_single_widget_layout(self) -> None:
        result = render_templates([{"widget": "bar_chart", "title": "Chart"}])
        entries = self._parse_layout(result["layout"])
        assert len(entries) == 1
        e = entries[0]
        assert e["i"] == 0
        assert e["x"] == 0
        assert e["w"] == 10
        assert e["minW"] == 5
        assert e["h"] == 10
        assert e["static"] is False

    def test_two_widgets_correct_count(self) -> None:
        result = render_templates(
            [
                {"widget": "bar_chart", "title": "Chart"},
                {"widget": "events_map", "title": "Map"},
            ]
        )
        entries = self._parse_layout(result["layout"])
        assert len(entries) == 2

    def test_widget_ids_are_sequential(self) -> None:
        widgets = [
            {"widget": "events_map", "title": "Map"},
            {"widget": "bar_chart", "title": "Chart"},
            {"widget": "pie_chart", "title": "Pie"},
        ]
        result = render_templates(widgets)
        entries = self._parse_layout(result["layout"])
        ids = [e["i"] for e in entries]
        assert ids == [0, 1, 2]

    def test_widget_ids_match_loop_order(self) -> None:
        """Widget at loop index 0 gets widget_id 0, etc."""
        widgets = [
            {"widget": "events_map", "title": "Map"},
            {"widget": "bar_chart", "title": "Chart"},
            {"widget": "pie_chart", "title": "Pie"},
            {"widget": "event_count_map", "title": "Heatmap"},
            {"widget": "events_table", "title": "Table"},
        ]
        result = render_templates(widgets)
        entries = self._parse_layout(result["layout"])
        assert len(entries) == 5
        for idx, entry in enumerate(entries):
            assert entry["i"] == idx

    def test_map_chart_pair_widths(self) -> None:
        """map left + chart right: map w=6, chart w=4."""
        result = render_templates(
            [
                {"widget": "events_map", "title": "Map"},
                {"widget": "bar_chart", "title": "Chart"},
            ]
        )
        entries = self._parse_layout(result["layout"])
        map_entry = entries[0]
        chart_entry = entries[1]
        assert map_entry["x"] == 0
        assert map_entry["w"] == 6
        assert map_entry["minW"] == 5
        assert chart_entry["x"] == 6
        assert chart_entry["w"] == 4
        assert chart_entry["minW"] == 4

    def test_chart_map_pair_widths(self) -> None:
        """chart left + map right: chart w=4, map w=6."""
        result = render_templates(
            [
                {"widget": "bar_chart", "title": "Chart"},
                {"widget": "events_map", "title": "Map"},
            ]
        )
        entries = self._parse_layout(result["layout"])
        chart_entry = entries[0]
        map_entry = entries[1]
        assert chart_entry["x"] == 0
        assert chart_entry["w"] == 4
        assert chart_entry["minW"] == 4
        assert map_entry["x"] == 4
        assert map_entry["w"] == 6
        assert map_entry["minW"] == 5

    def test_map_map_pair_widths(self) -> None:
        """map + map: each w=5."""
        result = render_templates(
            [
                {"widget": "events_map", "title": "Map 1"},
                {"widget": "event_count_map", "title": "Map 2"},
            ]
        )
        entries = self._parse_layout(result["layout"])
        assert entries[0]["x"] == 0
        assert entries[0]["w"] == 5
        assert entries[0]["minW"] == 5
        assert entries[1]["x"] == 5
        assert entries[1]["w"] == 5
        assert entries[1]["minW"] == 5

    def test_chart_chart_pair_widths(self) -> None:
        """chart + chart: each w=5."""
        result = render_templates(
            [
                {"widget": "bar_chart", "title": "Bar"},
                {"widget": "pie_chart", "title": "Pie"},
            ]
        )
        entries = self._parse_layout(result["layout"])
        assert entries[0]["x"] == 0
        assert entries[0]["w"] == 5
        assert entries[0]["minW"] == 4
        assert entries[1]["x"] == 5
        assert entries[1]["w"] == 5
        assert entries[1]["minW"] == 4

    def test_row_y_values(self) -> None:
        """Three widgets: first pair y=0, solo third widget y=10."""
        result = render_templates(
            [
                {"widget": "bar_chart", "title": "Chart 1"},
                {"widget": "pie_chart", "title": "Chart 2"},
                {"widget": "events_map", "title": "Map"},
            ]
        )
        entries = self._parse_layout(result["layout"])
        assert entries[0]["y"] == 0
        assert entries[1]["y"] == 0
        assert entries[2]["y"] == 10

    def test_five_widgets_two_rows(self) -> None:
        """5 widgets: pair in row 0, pair in row 1, solo in row 2."""
        widgets = [
            {"widget": "events_map", "title": "Map"},
            {"widget": "bar_chart", "title": "Chart"},
            {"widget": "pie_chart", "title": "Pie"},
            {"widget": "event_count_map", "title": "Heatmap"},
            {"widget": "events_table", "title": "Table"},
        ]
        result = render_templates(widgets)
        entries = self._parse_layout(result["layout"])
        assert entries[0]["y"] == 0
        assert entries[1]["y"] == 0
        assert entries[2]["y"] == 10
        assert entries[3]["y"] == 10
        assert entries[4]["y"] == 20

    def test_all_entries_height_10(self) -> None:
        widgets = [
            {"widget": "bar_chart", "title": "A"},
            {"widget": "pie_chart", "title": "B"},
            {"widget": "events_map", "title": "C"},
        ]
        result = render_templates(widgets)
        entries = self._parse_layout(result["layout"])
        for e in entries:
            assert e["h"] == 10

    def test_all_entries_static_false(self) -> None:
        result = render_templates([{"widget": "bar_chart", "title": "Chart"}])
        entries = self._parse_layout(result["layout"])
        for e in entries:
            assert e["static"] is False

    def test_layout_is_valid_json(self) -> None:
        widgets = [
            {"widget": "events_map", "title": "Map"},
            {"widget": "bar_chart", "title": "Chart"},
            {"widget": "pie_chart", "title": "Pie"},
        ]
        result = render_templates(widgets)
        parsed = json.loads(result["layout"])
        assert isinstance(parsed, list)
