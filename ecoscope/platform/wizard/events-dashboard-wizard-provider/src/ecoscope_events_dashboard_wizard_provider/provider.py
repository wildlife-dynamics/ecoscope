"""EcoscopeEventsDashboardWizardProvider — events dashboard workflow scaffolder.

Extends ``DefaultWizardProvider`` with per-widget include/title questions.
For each of the five available widget types the wizard asks two questions:

1. Include? (yes/no)
2. Display title (max 50 chars) — asked only when include = yes

The canonical order (bar_chart → events_map → pie_chart → event_count_map →
events_table) determines widget_id assignment in the generated ``layout.json``
and the order of entries in ``gather_dashboard.widgets``.

Examples:
    Drive the wizard with a sequence of answers::

        >>> from ecoscope_events_dashboard_wizard_provider import (
        ...     EcoscopeEventsDashboardWizardProvider,
        ... )
        >>> p = EcoscopeEventsDashboardWizardProvider()
        >>> qs = p.get_questions()
        >>> qs[-1]["dest"]
        'events_table_title'

"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from wt_compiler.wizard.abstract import (
    ArgparseKwargs,
    SingleWizardQuestion,
    WizardKwargs,
    WizardQuestion,
)
from wt_compiler.wizard.default import DefaultWizardProvider

WIDGET_CHOICES: list[str] = [
    "bar_chart",
    "events_map",
    "pie_chart",
    "event_count_map",
    "events_table",
]
"""Available dashboard widget types, in canonical display order."""

WIDGET_LABELS: dict[str, str] = {
    "bar_chart": "Bar Chart",
    "events_map": "Events Map",
    "pie_chart": "Pie Chart",
    "event_count_map": "Event Count Map",
    "events_table": "Events Table",
}
"""Human-readable labels for each widget type."""


# --- Validation callables ---------------------------------------------------


def widget_title_type(value: str) -> str:
    """Validate a widget display title.

    Must be non-empty after stripping whitespace and at most 50 characters.

    Args:
        value: The title string to validate.

    Returns:
        The stripped title string.

    Raises:
        argparse.ArgumentTypeError: If the title is empty or exceeds 50 characters.

    Examples:
        >>> widget_title_type("Events Map")
        'Events Map'
        >>> widget_title_type("  Events Map  ")
        'Events Map'
        >>> widget_title_type("")
        Traceback (most recent call last):
            ...
        argparse.ArgumentTypeError: Widget title cannot be empty.
        >>> widget_title_type("A" * 51)
        Traceback (most recent call last):
            ...
        argparse.ArgumentTypeError: Widget title must be 50 characters or fewer.
    """
    stripped = value.strip() if value else ""
    if not stripped:
        raise argparse.ArgumentTypeError("Widget title cannot be empty.")
    if len(stripped) > 50:
        raise argparse.ArgumentTypeError("Widget title must be 50 characters or fewer.")
    return stripped


# --- Widget question definitions --------------------------------------------


def _make_widget_questions() -> list[WizardQuestion]:
    """Build the include/title question pairs for all widget types.

    For each widget type in ``WIDGET_CHOICES`` two questions are created:

    1. An include question (``dest="include_<wtype>"``, choices yes/no).
    2. A title question (``dest="<wtype>_title"``) that is only asked when
       the include answer is ``"yes"``.

    Returns:
        Flat list of ``SingleWizardQuestion`` dicts, two per widget type.

    Examples:
        >>> qs = _make_widget_questions()
        >>> len(qs)
        10
        >>> qs[0]["dest"]
        'include_bar_chart'
        >>> qs[1]["dest"]
        'bar_chart_title'
    """
    questions: list[WizardQuestion] = []
    for wtype in WIDGET_CHOICES:
        dest_include = f"include_{wtype}"
        dest_title = f"{wtype}_title"
        label = WIDGET_LABELS[wtype]
        questions.append(
            SingleWizardQuestion(
                dest=dest_include,
                argparse=ArgparseKwargs(
                    help=f"Include {label} widget?",
                    choices=["yes", "no"],
                ),
                wizard=WizardKwargs(),
            )
        )
        questions.append(
            SingleWizardQuestion(
                dest=dest_title,
                argparse=ArgparseKwargs(
                    help=f"{label} display title (max 50 chars)",
                    type=widget_title_type,
                    default=label,
                ),
                wizard=WizardKwargs(
                    condition=lambda a, d=dest_include: a.get(d) == "yes",  # type: ignore[misc]
                ),
            )
        )
    return questions


_WIDGET_QUESTIONS: list[WizardQuestion] = _make_widget_questions()
"""Pre-built include/title question pairs for all widget types."""


# --- Provider ---------------------------------------------------------------


class EcoscopeEventsDashboardWizardProvider(DefaultWizardProvider):
    """Wizard provider for Ecoscope events dashboard workflow scaffolding.

    Extends ``DefaultWizardProvider`` with per-widget include/title questions.
    For each of the five widget types the wizard first asks whether to include
    the widget (yes/no), then — if yes — asks for a display title (max 50
    chars, defaults to the canonical label).

    At least one widget must be included; ``dump()`` raises ``ValueError``
    if no widgets are selected.

    Examples:
        >>> p = EcoscopeEventsDashboardWizardProvider()
        >>> questions = p.get_questions()
        >>> questions[-1]["dest"]
        'events_table_title'
        >>> questions[-2]["dest"]
        'include_events_table'
        >>> questions[-1]["argparse"]["default"]
        'Events Table'
    """

    def get_questions(self) -> list[WizardQuestion]:
        """Return default questions followed by the widget include/title pairs.

        Returns:
            All questions from ``DefaultWizardProvider`` plus two questions
            per widget type (include yes/no and conditional display title).
        """
        return [
            *super().get_questions(),
            *copy.deepcopy(_WIDGET_QUESTIONS),
        ]

    def dump(self, workdir: Path) -> None:
        """Render templates to *workdir* after validating widget selection.

        Validates that at least one widget is included before delegating to
        the parent ``dump()``.

        Args:
            workdir: Directory to write rendered files into.

        Raises:
            ValueError: If no widgets are selected.
            jinja2.UndefinedError: If answers are missing for a template variable.
        """
        enabled = [w for w in WIDGET_CHOICES if self._answers.get(f"include_{w}") == "yes"]
        if not enabled:
            raise ValueError(
                "At least one widget must be selected. " "Re-run the wizard and answer 'yes' to at least one widget."
            )
        super().dump(workdir)
