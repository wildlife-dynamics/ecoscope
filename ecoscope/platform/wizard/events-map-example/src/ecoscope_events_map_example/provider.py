"""EcoscopeEventsMapExampleProvider — events map workflow scaffolder.

Extends ``DefaultWizardProvider`` with a single title question for the events
map widget.

Examples:
    Drive the wizard with a sequence of answers::

        >>> from ecoscope_events_map_example import EcoscopeEventsMapExampleProvider
        >>> p = EcoscopeEventsMapExampleProvider()
        >>> qs = p.get_questions()
        >>> qs[-1]["dest"]
        'events_map_title'

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

MAP_TITLE_DEFAULT: str = "Events Map"
"""Default display title for the events map widget."""


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


# --- Question definition ----------------------------------------------------

_Q_EVENTS_MAP_TITLE: SingleWizardQuestion = SingleWizardQuestion(
    dest="events_map_title",
    argparse=ArgparseKwargs(
        help="Events map display title (max 50 chars)",
        type=widget_title_type,
        default=MAP_TITLE_DEFAULT,
    ),
    wizard=WizardKwargs(),
)
"""Title question for the events map widget."""


# --- Provider ---------------------------------------------------------------


class EcoscopeEventsMapExampleProvider(DefaultWizardProvider):
    """Wizard provider for Ecoscope events map workflow scaffolding.

    Extends ``DefaultWizardProvider`` with a single title question for the
    events map widget.

    Examples:
        >>> p = EcoscopeEventsMapExampleProvider()
        >>> questions = p.get_questions()
        >>> questions[-1]["dest"]
        'events_map_title'
        >>> questions[-1]["argparse"]["default"]
        'Events Map'
    """

    def get_questions(self) -> list[WizardQuestion]:
        """Return default questions (minus requirements) plus the map title question.

        The requirements loop is omitted; requirements are fixed in the template.

        Returns:
            Default questions without ``requirements``, plus the events map
            title question.
        """
        return [q for q in super().get_questions() if q["dest"] != "requirements"] + [
            copy.deepcopy(_Q_EVENTS_MAP_TITLE)
        ]

    def dump(self, workdir: Path) -> None:
        """Render templates, injecting an empty requirements list for the template.

        Args:
            workdir: Directory to write rendered files into.
        """
        self._answers.setdefault("requirements", [])
        super().dump(workdir)
