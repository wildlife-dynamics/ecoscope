"""Tests for EcoscopeEventsMapExampleProvider and associated validators."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pytest
from ecoscope_events_map_example import EcoscopeEventsMapExampleProvider
from ecoscope_events_map_example.provider import (
    MAP_TITLE_DEFAULT,
    widget_title_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def drive_wizard(
    provider: EcoscopeEventsMapExampleProvider,
    answers: list[str | None],
) -> list[dict]:
    """Drive wizard generator with a sequence of answers.

    Returns all yielded questions.
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


def make_provider(title: str = MAP_TITLE_DEFAULT) -> EcoscopeEventsMapExampleProvider:
    """Drive a provider through all questions and return it with answers populated."""
    provider = EcoscopeEventsMapExampleProvider()
    answers: list[str | None] = [
        "my_dashboard",  # workflow_id
        "My Dashboard",  # workflow_name
        "Dashboard description",  # workflow_description
        "Test Author",  # author_name
        "MIT",  # license_type
        title,  # events_map_title
    ]
    drive_wizard(provider, answers)
    return provider


def render_templates(title: str = MAP_TITLE_DEFAULT) -> dict[str, str]:
    """Render spec.yaml and layout.json for the given title.

    Returns:
        Dict with keys ``spec`` and ``layout`` containing rendered file text.
    """
    provider = make_provider(title)
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

    def test_last_question_is_events_map_title(self) -> None:
        p = EcoscopeEventsMapExampleProvider()
        questions = p.get_questions()
        assert questions[-1]["dest"] == "events_map_title"

    def test_question_count(self) -> None:
        from wt_compiler.wizard.default import DefaultWizardProvider

        # default minus requirements, plus events_map_title
        default_count = len(DefaultWizardProvider().get_questions())
        provider_count = len(EcoscopeEventsMapExampleProvider().get_questions())
        assert provider_count == default_count  # -1 requirements, +1 title

    def test_requirements_question_absent(self) -> None:
        p = EcoscopeEventsMapExampleProvider()
        dests = [q["dest"] for q in p.get_questions()]
        assert "requirements" not in dests

    def test_title_has_correct_default(self) -> None:
        p = EcoscopeEventsMapExampleProvider()
        questions = p.get_questions()
        assert questions[-1]["argparse"]["default"] == MAP_TITLE_DEFAULT

    def test_inherits_default_questions_except_requirements(self) -> None:
        from wt_compiler.wizard.default import DefaultWizardProvider

        default_dests = [q["dest"] for q in DefaultWizardProvider().get_questions()]
        provider_dests = [q["dest"] for q in EcoscopeEventsMapExampleProvider().get_questions()]
        for dest in default_dests:
            if dest == "requirements":
                assert dest not in provider_dests
            else:
                assert dest in provider_dests

    def test_get_questions_returns_independent_copies(self) -> None:
        """Each call to get_questions() returns a fresh deep copy."""
        p = EcoscopeEventsMapExampleProvider()
        q1 = p.get_questions()
        q2 = p.get_questions()
        assert q1[-1] is not q2[-1]


# ---------------------------------------------------------------------------
# dump()
# ---------------------------------------------------------------------------


class TestDumpValidation:
    """Tests for dump() output."""

    def test_dump_writes_spec_and_layout(self, tmp_path: Path) -> None:
        provider = make_provider()
        provider.dump(tmp_path)
        assert (tmp_path / "spec.yaml").exists()
        assert (tmp_path / "layout.json").exists()

    def test_dump_uses_custom_title(self, tmp_path: Path) -> None:
        provider = make_provider(title="Custom Title")
        provider.dump(tmp_path)
        spec = (tmp_path / "spec.yaml").read_text()
        assert "Custom Title" in spec


# ---------------------------------------------------------------------------
# Template rendering — spec.yaml
# ---------------------------------------------------------------------------


class TestSpecRendering:
    """Tests for spec.yaml template rendering."""

    def test_events_map_title_default_in_spec(self) -> None:
        result = render_templates()
        assert MAP_TITLE_DEFAULT in result["spec"]

    def test_custom_title_in_spec(self) -> None:
        result = render_templates(title="My Custom Map")
        assert "My Custom Map" in result["spec"]

    def test_workflow_id_in_spec(self) -> None:
        result = render_templates()
        assert "my_dashboard" in result["spec"]

    def test_wt_compiler_references_not_evaluated(self) -> None:
        result = render_templates()
        assert "${{ workflow." in result["spec"]


# ---------------------------------------------------------------------------
# Template rendering — layout.json
# ---------------------------------------------------------------------------


class TestLayoutRendering:
    """Tests for layout.json template rendering."""

    def _parse_layout(self, layout_text: str) -> list[dict]:
        return json.loads(layout_text)

    def test_layout_has_single_entry(self) -> None:
        result = render_templates()
        entries = self._parse_layout(result["layout"])
        assert len(entries) == 1

    def test_layout_entry_is_full_width(self) -> None:
        result = render_templates()
        entry = self._parse_layout(result["layout"])[0]
        assert entry["i"] == 0
        assert entry["x"] == 0
        assert entry["y"] == 0
        assert entry["w"] == 10
        assert entry["h"] == 10
        assert entry["minW"] == 5
        assert entry["static"] is False

    def test_layout_is_valid_json(self) -> None:
        result = render_templates()
        parsed = json.loads(result["layout"])
        assert isinstance(parsed, list)
