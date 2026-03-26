"""Tests for EcoscopeEventsDashboardWizardProvider and associated validators."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pytest
from ecoscope_events_dashboard_wizard_provider import EcoscopeEventsDashboardWizardProvider
from ecoscope_events_dashboard_wizard_provider.provider import (
    WIDGET_CHOICES,
    WIDGET_LABELS,
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
    enabled: dict[str, str],
) -> EcoscopeEventsDashboardWizardProvider:
    """Drive a provider through default questions and per-widget include/title questions.

    Args:
        enabled: Mapping of widget type to display title for widgets that
            should be included.  Widget types absent from the mapping are
            answered ``"no"`` and their title question is skipped.

    Returns:
        A provider whose answers include all default fields and the widget
        include/title answers.
    """
    provider = EcoscopeEventsDashboardWizardProvider()
    base_answers: list[str | None] = [
        "my_dashboard",  # workflow_id
        "My Dashboard",  # workflow_name
        "Dashboard description",  # workflow_description
        "Test Author",  # author_name
        "MIT",  # license_type
        None,  # terminate requirements loop
    ]
    widget_answers: list[str | None] = []
    for wtype in WIDGET_CHOICES:
        if wtype in enabled:
            widget_answers.append("yes")  # include_<wtype>
            widget_answers.append(enabled[wtype])  # <wtype>_title
        else:
            widget_answers.append("no")  # include_<wtype>; title skipped
    drive_wizard(provider, base_answers + widget_answers)
    return provider


def render_templates(enabled: dict[str, str]) -> dict[str, str]:
    """Render spec.yaml and layout.json for the given enabled widgets.

    Args:
        enabled: Mapping of widget type to display title for enabled widgets.

    Returns:
        Dict with keys ``spec`` and ``layout`` containing rendered file text.
    """
    provider = make_provider_with_widgets(enabled)
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
# Question structure
# ---------------------------------------------------------------------------


class TestQuestionStructure:
    """Tests for provider question list structure."""

    def test_last_question_is_events_table_title(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        assert questions[-1]["dest"] == "events_table_title"

    def test_second_to_last_is_include_events_table(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        assert questions[-2]["dest"] == "include_events_table"

    def test_ten_widget_questions_appended(self) -> None:
        from wt_compiler.wizard.default import DefaultWizardProvider

        default_count = len(DefaultWizardProvider().get_questions())
        provider_count = len(EcoscopeEventsDashboardWizardProvider().get_questions())
        assert provider_count == default_count + 10

    def test_include_questions_have_yes_no_choices(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        include_qs = [q for q in questions if q["dest"].startswith("include_")]
        assert len(include_qs) == 5
        for q in include_qs:
            assert q["argparse"]["choices"] == ["yes", "no"]

    def test_title_questions_have_default_labels(self) -> None:
        p = EcoscopeEventsDashboardWizardProvider()
        questions = p.get_questions()
        for wtype, label in WIDGET_LABELS.items():
            title_q = next(q for q in questions if q["dest"] == f"{wtype}_title")
            assert title_q["argparse"]["default"] == label

    def test_title_question_condition_skipped_when_no(self) -> None:
        """Title question is not yielded when include answer is 'no'."""
        provider = EcoscopeEventsDashboardWizardProvider()
        base_answers: list[str | None] = [
            "my_wf",
            "My WF",
            "desc",
            "Author",
            "MIT",
            None,
        ]
        # Answer "no" to all include questions
        widget_answers: list[str | None] = ["no"] * len(WIDGET_CHOICES)
        questions = drive_wizard(provider, base_answers + widget_answers)
        yielded_dests = {q["dest"] for q in questions}
        for wtype in WIDGET_CHOICES:
            assert f"{wtype}_title" not in yielded_dests

    def test_title_question_yielded_when_yes(self) -> None:
        """Title question is yielded when include answer is 'yes'."""
        provider = EcoscopeEventsDashboardWizardProvider()
        base_answers: list[str | None] = [
            "my_wf",
            "My WF",
            "desc",
            "Author",
            "MIT",
            None,
        ]
        # Answer "yes" only for bar_chart; "no" for rest
        widget_answers: list[str | None] = [
            "yes",
            "My Chart",  # bar_chart
            "no",  # events_map
            "no",  # pie_chart
            "no",  # event_count_map
            "no",  # events_table
        ]
        questions = drive_wizard(provider, base_answers + widget_answers)
        yielded_dests = {q["dest"] for q in questions}
        assert "bar_chart_title" in yielded_dests
        for wtype in ("events_map", "pie_chart", "event_count_map", "events_table"):
            assert f"{wtype}_title" not in yielded_dests

    def test_inherits_default_questions(self) -> None:
        from wt_compiler.wizard.default import DefaultWizardProvider

        default = DefaultWizardProvider()
        provider = EcoscopeEventsDashboardWizardProvider()
        default_dests = [q["dest"] for q in default.get_questions()]
        provider_dests = [q["dest"] for q in provider.get_questions()]
        for dest in default_dests:
            assert dest in provider_dests

    def test_get_questions_returns_independent_copies(self) -> None:
        """Each call to get_questions() returns fresh deep copies."""
        p = EcoscopeEventsDashboardWizardProvider()
        q1 = p.get_questions()
        q2 = p.get_questions()
        assert q1[-1] is not q2[-1]


# ---------------------------------------------------------------------------
# dump() validation
# ---------------------------------------------------------------------------


class TestDumpValidation:
    """Tests for dump() widget validation."""

    def test_dump_raises_on_no_widgets_selected(self, tmp_path: Path) -> None:
        """dump() raises ValueError when all widgets are answered 'no'."""
        provider = EcoscopeEventsDashboardWizardProvider()
        base_answers: list[str | None] = [
            "my_dashboard",
            "My Dashboard",
            "Dashboard description",
            "Test Author",
            "MIT",
            None,
        ]
        widget_answers: list[str | None] = ["no"] * len(WIDGET_CHOICES)
        drive_wizard(provider, base_answers + widget_answers)
        with pytest.raises(ValueError, match="[Aa]t least one widget"):
            provider.dump(tmp_path)

    def test_dump_succeeds_with_one_widget(self, tmp_path: Path) -> None:
        """dump() succeeds and writes files when exactly one widget is selected."""
        provider = make_provider_with_widgets({"bar_chart": "Events"})
        provider.dump(tmp_path)
        assert (tmp_path / "spec.yaml").exists()
        assert (tmp_path / "layout.json").exists()


# ---------------------------------------------------------------------------
# Template rendering — spec.yaml
# ---------------------------------------------------------------------------


class TestSpecRendering:
    """Tests for spec.yaml template rendering."""

    def test_bar_chart_tasks_present_when_enabled(self) -> None:
        result = render_templates({"bar_chart": "My Bar Chart"})
        assert "events_bar_chart" in result["spec"]
        assert "grouped_bar_plot_widget_merge" in result["spec"]

    def test_events_map_tasks_present_when_enabled(self) -> None:
        result = render_templates({"events_map": "My Map"})
        assert "grouped_events_map_layer" in result["spec"]
        assert "grouped_events_map_widget_merge" in result["spec"]

    def test_pie_chart_tasks_present_when_enabled(self) -> None:
        result = render_templates({"pie_chart": "My Pie"})
        assert "grouped_events_pie_chart" in result["spec"]
        assert "grouped_events_pie_widget_merge" in result["spec"]

    def test_event_count_map_tasks_present_when_enabled(self) -> None:
        result = render_templates({"event_count_map": "Heatmap"})
        assert "events_meshgrid" in result["spec"]
        assert "grouped_fd_map_widget_merge" in result["spec"]

    def test_events_table_tasks_present_when_enabled(self) -> None:
        result = render_templates({"events_table": "Event List"})
        assert "events_table" in result["spec"]
        assert "grouped_table_widget_merge" in result["spec"]

    def test_disabled_widget_tasks_absent(self) -> None:
        """Only bar_chart selected — other widget tasks must not appear."""
        result = render_templates({"bar_chart": "Chart"})
        assert "events_meshgrid" not in result["spec"]
        assert "grouped_events_pie_chart" not in result["spec"]
        assert "grouped_events_map_layer" not in result["spec"]
        assert "grouped_table_widget_merge" not in result["spec"]

    def test_custom_title_appears_in_set_string_var(self) -> None:
        result = render_templates({"bar_chart": "My Custom Title"})
        assert "My Custom Title" in result["spec"]

    def test_gather_dashboard_widgets_in_canonical_order(self) -> None:
        """With bar_chart and events_map enabled, bar_chart appears first."""
        result = render_templates({"bar_chart": "Bar Chart", "events_map": "Events Map"})
        spec = result["spec"]
        gather_pos = spec.index("gather_dashboard")
        bar_in_gather = spec.index("grouped_bar_plot_widget_merge", gather_pos)
        map_in_gather = spec.index("grouped_events_map_widget_merge", gather_pos)
        assert bar_in_gather < map_in_gather

    def test_all_five_widgets_render_all_tasks(self) -> None:
        result = render_templates(
            {
                "bar_chart": "Bar Chart",
                "events_map": "Events Map",
                "pie_chart": "Pie Chart",
                "event_count_map": "Heatmap",
                "events_table": "Table",
            }
        )
        spec = result["spec"]
        assert "grouped_bar_plot_widget_merge" in spec
        assert "grouped_events_map_widget_merge" in spec
        assert "grouped_events_pie_widget_merge" in spec
        assert "grouped_fd_map_widget_merge" in spec
        assert "grouped_table_widget_merge" in spec

    def test_rename_display_columns_present_for_map(self) -> None:
        result = render_templates({"events_map": "Map"})
        assert "rename_display_columns" in result["spec"]

    def test_rename_display_columns_present_for_table(self) -> None:
        result = render_templates({"events_table": "Table"})
        assert "rename_display_columns" in result["spec"]

    def test_rename_display_columns_absent_for_bar_only(self) -> None:
        result = render_templates({"bar_chart": "Chart"})
        assert "rename_display_columns" not in result["spec"]

    def test_base_map_defs_task_present_for_map_widget(self) -> None:
        result = render_templates({"events_map": "Map"})
        assert "set_base_maps" in result["spec"]

    def test_base_map_defs_task_present_for_event_count_map(self) -> None:
        result = render_templates({"event_count_map": "Heatmap"})
        assert "set_base_maps" in result["spec"]

    def test_base_map_defs_task_absent_when_no_map_widgets(self) -> None:
        result = render_templates({"bar_chart": "Chart", "pie_chart": "Pie"})
        assert "set_base_maps" not in result["spec"]

    def test_wt_compiler_references_not_evaluated(self) -> None:
        result = render_templates({"bar_chart": "Chart"})
        assert "${{ workflow." in result["spec"]

    def test_workflow_id_in_spec(self) -> None:
        result = render_templates({"bar_chart": "Chart"})
        assert "my_dashboard" in result["spec"]

    def test_fd_uischema_absent_when_event_count_map_disabled(self) -> None:
        result = render_templates({"bar_chart": "Chart"})
        assert "auto_scale_or_custom_cell_size" not in result["spec"]

    def test_fd_uischema_present_when_event_count_map_enabled(self) -> None:
        result = render_templates({"event_count_map": "Heatmap"})
        assert "auto_scale_or_custom_cell_size" in result["spec"]


# ---------------------------------------------------------------------------
# Template rendering — layout.json
# ---------------------------------------------------------------------------


class TestLayoutRendering:
    """Tests for layout.json template rendering."""

    def _parse_layout(self, layout_text: str) -> list[dict]:
        return json.loads(layout_text)

    def test_single_widget_layout(self) -> None:
        result = render_templates({"bar_chart": "Chart"})
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
        result = render_templates({"bar_chart": "Chart", "events_map": "Map"})
        entries = self._parse_layout(result["layout"])
        assert len(entries) == 2

    def test_widget_ids_are_sequential(self) -> None:
        result = render_templates(
            {
                "bar_chart": "Chart",
                "events_map": "Map",
                "pie_chart": "Pie",
            }
        )
        entries = self._parse_layout(result["layout"])
        assert [e["i"] for e in entries] == [0, 1, 2]

    def test_widget_ids_canonical_order(self) -> None:
        """Widget ids follow canonical order regardless of dict insertion order."""
        result = render_templates(
            {
                "events_table": "Table",
                "bar_chart": "Chart",
                "event_count_map": "Heatmap",
                "pie_chart": "Pie",
                "events_map": "Map",
            }
        )
        entries = self._parse_layout(result["layout"])
        assert len(entries) == 5
        assert [e["i"] for e in entries] == [0, 1, 2, 3, 4]

    def test_map_chart_pair_widths(self) -> None:
        """bar_chart (canonical pos 0) + events_map (pos 1): chart left, map right."""
        result = render_templates({"bar_chart": "Chart", "events_map": "Map"})
        entries = self._parse_layout(result["layout"])
        chart_entry = entries[0]  # bar_chart is canonical position 0
        map_entry = entries[1]  # events_map is canonical position 1
        assert chart_entry["x"] == 0
        assert chart_entry["w"] == 4
        assert chart_entry["minW"] == 4
        assert map_entry["x"] == 4
        assert map_entry["w"] == 6
        assert map_entry["minW"] == 5

    def test_map_map_pair_widths(self) -> None:
        """events_map + event_count_map: each w=5."""
        result = render_templates({"events_map": "Map 1", "event_count_map": "Map 2"})
        entries = self._parse_layout(result["layout"])
        assert entries[0]["x"] == 0
        assert entries[0]["w"] == 5
        assert entries[0]["minW"] == 5
        assert entries[1]["x"] == 5
        assert entries[1]["w"] == 5
        assert entries[1]["minW"] == 5

    def test_chart_chart_pair_widths(self) -> None:
        """bar_chart + pie_chart: each w=5."""
        result = render_templates({"bar_chart": "Bar", "pie_chart": "Pie"})
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
            {
                "bar_chart": "Chart 1",
                "pie_chart": "Chart 2",
                "events_map": "Map",
            }
        )
        entries = self._parse_layout(result["layout"])
        assert entries[0]["y"] == 0
        assert entries[1]["y"] == 0
        assert entries[2]["y"] == 10

    def test_five_widgets_three_rows(self) -> None:
        """5 widgets: pair row 0, pair row 1, solo row 2."""
        result = render_templates(
            {
                "bar_chart": "Chart",
                "events_map": "Map",
                "pie_chart": "Pie",
                "event_count_map": "Heatmap",
                "events_table": "Table",
            }
        )
        entries = self._parse_layout(result["layout"])
        assert entries[0]["y"] == 0
        assert entries[1]["y"] == 0
        assert entries[2]["y"] == 10
        assert entries[3]["y"] == 10
        assert entries[4]["y"] == 20

    def test_all_entries_height_10(self) -> None:
        result = render_templates(
            {
                "bar_chart": "A",
                "pie_chart": "B",
                "events_map": "C",
            }
        )
        entries = self._parse_layout(result["layout"])
        for e in entries:
            assert e["h"] == 10

    def test_all_entries_static_false(self) -> None:
        result = render_templates({"bar_chart": "Chart"})
        entries = self._parse_layout(result["layout"])
        for e in entries:
            assert e["static"] is False

    def test_layout_is_valid_json(self) -> None:
        result = render_templates(
            {
                "events_map": "Map",
                "bar_chart": "Chart",
                "pie_chart": "Pie",
            }
        )
        parsed = json.loads(result["layout"])
        assert isinstance(parsed, list)
