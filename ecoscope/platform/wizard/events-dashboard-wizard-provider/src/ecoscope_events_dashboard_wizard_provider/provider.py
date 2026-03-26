"""EcoscopeEventsDashboardWizardProvider — events dashboard workflow scaffolder.

Extends ``DefaultWizardProvider`` with a widget configuration loop that
collects dashboard widget types and display titles in user-defined order.
The loop order determines widget_id assignment in the generated ``layout.json``.

Examples:
    Drive the wizard with a sequence of answers::

        >>> from ecoscope_events_dashboard_wizard_provider import (
        ...     EcoscopeEventsDashboardWizardProvider,
        ... )
        >>> p = EcoscopeEventsDashboardWizardProvider()
        >>> qs = p.get_questions()
        >>> qs[-1]["dest"]
        'widgets'
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from wt_compiler.wizard.abstract import (
    ArgparseKwargs,
    SingleWizardQuestion,
    WizardKwargs,
    WizardQuestion,
    WizardQuestionLoop,
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


def _widget_batch_type(value: str) -> dict[str, Any]:
    """Parse and validate a single widget JSON object for batch mode.

    Expects a JSON object with ``widget`` (one of ``WIDGET_CHOICES``) and
    ``title`` (non-empty string, max 50 chars).

    Args:
        value: JSON string representing a single widget configuration.

    Returns:
        Validated dict with ``widget`` (str) and ``title`` (str) keys.

    Raises:
        argparse.ArgumentTypeError: On JSON parse failure or invalid fields.

    Examples:
        >>> d = _widget_batch_type('{"widget": "bar_chart", "title": "My Bar Chart"}')
        >>> d["widget"]
        'bar_chart'
        >>> d["title"]
        'My Bar Chart'
        >>> _widget_batch_type('{"widget": "unknown", "title": "X"}')
        Traceback (most recent call last):
            ...
        argparse.ArgumentTypeError: Invalid widget 'unknown': must be one of ...
    """
    import json

    try:
        d: dict[str, Any] = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}") from e
    if not isinstance(d, dict):
        raise argparse.ArgumentTypeError(f"Expected a JSON object (got {type(d).__name__})")

    widget = d.get("widget")
    if widget is None:
        raise argparse.ArgumentTypeError("Missing required field: widget")
    if widget not in WIDGET_CHOICES:
        raise argparse.ArgumentTypeError(f"Invalid widget '{widget}': must be one of {WIDGET_CHOICES}")
    d["widget"] = str(widget)

    title = d.get("title")
    if title is None:
        raise argparse.ArgumentTypeError("Missing required field: title")
    try:
        d["title"] = widget_title_type(str(title))
    except argparse.ArgumentTypeError as e:
        raise argparse.ArgumentTypeError(f"Invalid title: {e}") from e

    return d


# --- Widget loop question definition ----------------------------------------

_Q_WIDGETS_LOOP = WizardQuestionLoop(  # type: ignore[typeddict-unknown-key]
    dest="widgets",
    argparse=ArgparseKwargs(
        action="append",
        default=None,
        help=(
            'Widget as JSON: {"widget": "<type>", "title": "<display title>"}. '
            f"Available types: {', '.join(WIDGET_CHOICES)}. "
            "Repeat flag to add multiple widgets; order determines layout position."
        ),
        type=_widget_batch_type,
    ),
    questions=[
        # Sentinel question — MUST NOT carry wizard.condition
        SingleWizardQuestion(
            dest="widget",
            argparse=ArgparseKwargs(
                help="Widget type",
                choices=WIDGET_CHOICES,
            ),
            wizard=WizardKwargs(),
        ),
        SingleWizardQuestion(
            dest="title",
            argparse=ArgparseKwargs(
                help="Widget display title (max 50 chars)",
                type=widget_title_type,
            ),
            wizard=WizardKwargs(),
        ),
    ],
)


# --- Provider ---------------------------------------------------------------


class EcoscopeEventsDashboardWizardProvider(DefaultWizardProvider):
    """Wizard provider for Ecoscope events dashboard workflow scaffolding.

    Extends ``DefaultWizardProvider`` with a ``widgets`` loop question.
    Each loop iteration collects a widget type and display title.  The
    insertion order determines the widget_id assignment in ``layout.json``
    and the order of entries in ``gather_dashboard.widgets``.

    At least one widget must be selected; ``dump()`` raises ``ValueError``
    if the widgets list is empty or contains duplicate widget types.

    Examples:
        >>> p = EcoscopeEventsDashboardWizardProvider()
        >>> questions = p.get_questions()
        >>> questions[-1]["dest"]
        'widgets'
        >>> questions[-1]["questions"][0]["dest"]
        'widget'
        >>> questions[-1]["questions"][0]["argparse"]["choices"]  # doctest: +NORMALIZE_WHITESPACE
        ['bar_chart', 'events_map', 'pie_chart', 'event_count_map', 'events_table']
    """

    def get_questions(self) -> list[WizardQuestion]:
        """Return default questions followed by the widget configuration loop.

        Returns:
            All questions from ``DefaultWizardProvider`` plus the ``widgets``
            ``WizardQuestionLoop``.
        """
        return [
            *super().get_questions(),
            copy.deepcopy(_Q_WIDGETS_LOOP),
        ]

    def dump(self, workdir: Path) -> None:
        """Render templates to *workdir* after validating widget configuration.

        Validates that at least one widget is selected and that no duplicate
        widget types are present before delegating to the parent ``dump()``.

        Args:
            workdir: Directory to write rendered files into.

        Raises:
            ValueError: If no widgets are selected or duplicate widget types
                are present.
            jinja2.UndefinedError: If answers are missing for a template variable.
        """
        widgets: list[dict[str, Any]] = self._answers.get("widgets") or []
        if not widgets:
            raise ValueError(
                "At least one widget must be selected. "
                "Re-run the wizard and add at least one widget to the dashboard."
            )
        seen: set[str] = set()
        for entry in widgets:
            widget_type = entry["widget"]
            if widget_type in seen:
                raise ValueError(f"Duplicate widget type '{widget_type}'. " "Each widget type may only be added once.")
            seen.add(widget_type)
        super().dump(workdir)
