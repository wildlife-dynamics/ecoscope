"""JSON schema utilities for React JSON Schema Form (RJSF) integration.

Provides models for building filter schemas compatible with RJSF, including
``oneOf`` enum fields, filter property/UI-schema pairs, and a container that
generates combined JSON schemas for multiple named filters.
"""

from typing import Any, Literal

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
