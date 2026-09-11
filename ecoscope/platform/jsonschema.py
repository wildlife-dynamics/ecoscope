from typing import Any, Literal, get_args

from pydantic import BaseModel, model_serializer


class oneOf(BaseModel):
    """Model representing the oneOf field in a JSON schema.

    Args:
        const: The value that will appear in the form data.
        title: The user-facing name that will appear in the input widget.
    """

    const: Any
    title: str


class RJSFFilterProperty(BaseModel):
    """Model representing the properties of a React JSON Schema Form filter.
    This model is used to generate the `properties` field for a filter schema in a dashboard.

    Args:
        type: The type of the filter property.
        oneOf: The possible values for the filter property.
        default: The default value for the filter property
    """

    type: str
    title: str
    oneOf: list[oneOf]
    default: str


class RJSFFilterUiSchema(BaseModel):
    """Model representing the UI schema of a React JSON Schema Form filter.
    This model is used to generate the `uiSchema` field for a filter schema in a dashboard.

    Args:
        title: The title of the filter.
        help: The help text for the filter.
    """

    title: str
    help: str | None = None
    widget: Literal["select"] = "select"

    @model_serializer
    def ser_model(self) -> dict[str, Any]:
        return {
            "ui:title": self.title,
            "ui:widget": self.widget,
        } | ({"ui:help": self.help} if self.help else {})


class RJSFFilter(BaseModel):
    """Model representing a React JSON Schema Form filter."""

    property: RJSFFilterProperty
    uiSchema: RJSFFilterUiSchema


class ReactJSONSchemaFormFilters(BaseModel):
    options: dict[str, RJSFFilter]

    @property
    def _schema(self):
        return {
            "type": "object",
            "properties": {opt: rjsf.property.model_dump() for opt, rjsf in self.options.items()},
            "uiSchema": {opt: rjsf.uiSchema.model_dump() for opt, rjsf in self.options.items()},
        }

    @model_serializer
    def ser_model(self) -> dict[str, Any]:
        return {"schema": self._schema}


def labeled_literal_items(literal: Any, labels: dict[str, str] | None = None) -> dict[str, Any]:
    """`json_schema_extra` for a `list[Literal[...]]` field: render labeled options.

    A `list[Literal[...]]` puts its bare `enum` inside `items`, so the widget shows the
    wire values verbatim. Replacing `items` wholesale with `oneOf` const/title pairs
    labels it without changing what it submits:

    >>> labeled_literal_items(Literal["new", "in_review"])["items"]["oneOf"]
    [{'const': 'new', 'title': 'New'}, {'const': 'in_review', 'title': 'In Review'}]

    Options come from the literal itself, so they cannot drift from what validation
    accepts. Titles default to the title-cased value; `labels` overrides any of them:

    >>> labeled_literal_items(Literal["new"], {"new": "Fresh"})["items"]["oneOf"]
    [{'const': 'new', 'title': 'Fresh'}]

    The return is a plain dict rather than the callable `json_schema_extra` form
    `labeled_units` uses, so it composes with `AdvancedField`, which merges its
    always-set keys by dict union and would raise `TypeError` on a callable.
    """
    labels = labels or {}
    return {
        "items": {
            "type": "string",
            "oneOf": [
                oneOf(const=value, title=labels.get(value, value.replace("_", " ").title())).model_dump()
                for value in get_args(literal)
            ],
        }
    }
