from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError
from pydantic.json_schema import SkipJsonSchema

from ecoscope.platform.annotations import AdvancedField
from ecoscope.platform.jsonschema import labeled_literal_items

Colour = Literal["red", "burnt_orange"]


def test_labeled_literal_items_derives_options_from_the_literal():
    assert labeled_literal_items(Colour) == {
        "items": {
            "type": "string",
            "oneOf": [
                {"const": "red", "title": "Red"},
                {"const": "burnt_orange", "title": "Burnt Orange"},
            ],
        }
    }


def test_labeled_literal_items_honours_explicit_labels():
    extra = labeled_literal_items(Colour, {"red": "Rogue"})
    titles = [option["title"] for option in extra["items"]["oneOf"]]
    # Unlabelled members still fall back to the title-cased value.
    assert titles == ["Rogue", "Burnt Orange"]


def test_labeled_literal_items_replaces_the_item_enum():
    class M(BaseModel):
        colours: Annotated[
            list[Colour] | SkipJsonSchema[None],
            Field(default=None, json_schema_extra={"uniqueItems": True, **labeled_literal_items(Colour)}),
        ] = None

    items = M.model_json_schema()["properties"]["colours"]["items"]
    # The bare enum is gone, so the widget renders titles rather than wire values.
    assert "enum" not in items
    assert [option["const"] for option in items["oneOf"]] == ["red", "burnt_orange"]


def test_labeled_literal_items_composes_with_advanced_field():
    # AdvancedField merges its always-set keys by dict union, so a callable
    # json_schema_extra (the `labeled_units` form) would raise TypeError here.
    field = AdvancedField(default=None, json_schema_extra={**labeled_literal_items(Colour)})
    extra = field.json_schema_extra
    assert extra["ecoscope:advanced"] is True
    assert extra["items"]["oneOf"][0] == {"const": "red", "title": "Red"}


def test_labeling_does_not_change_accepted_values():
    class M(BaseModel):
        colours: Annotated[
            list[Colour] | SkipJsonSchema[None],
            Field(default=None, json_schema_extra=labeled_literal_items(Colour)),
        ] = None

    # `const` is what the form submits; `title` is display only.
    assert M(colours=["burnt_orange"]).colours == ["burnt_orange"]
    with pytest.raises(ValidationError):
        M(colours=["Burnt Orange"])
