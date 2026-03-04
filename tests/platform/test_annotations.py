from dataclasses import dataclass
from typing import Annotated

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pa_typing
import pytest
from ecoscope.platform.annotations import (
    AdvancedField,
    DataFrame,
    EmptyDataFrameModel,
    JsonSerializableDataFrameModel,
)
from pydantic import BaseModel, Field, ValidationError
from wt_registry.jsonschema import jsonschema_from_task_func


def test_advanced_field_task_annotation():
    def f(
        basic_field: Annotated[str, Field()],
        advanced_field: Annotated[str, AdvancedField(default=None)],
    ):
        pass

    schema = jsonschema_from_task_func(f)
    assert set(schema["properties"]) == {"basic_field", "advanced_field"}
    assert set(schema["required"]) == {"basic_field"}
    assert schema["properties"]["advanced_field"]["ecoscope:advanced"]


def test_advanced_field_nested_basemodel_annotated():
    class NestedModel(BaseModel):
        basic_field: Annotated[str, Field()]
        advanced_field: Annotated[str, AdvancedField(default=None)]

    def f(nested_model: Annotated[NestedModel, Field()]):
        pass

    schema = jsonschema_from_task_func(f)
    nested_model_def = schema["$defs"]["NestedModel"]
    assert set(nested_model_def["properties"]) == {"basic_field", "advanced_field"}
    assert set(nested_model_def["required"]) == {"basic_field"}


def test_advanced_field_nested_basemodel_not_annotated():
    class NestedModel(BaseModel):
        basic_field: str = Field()
        advanced_field: str = AdvancedField(default=None)

    def f(nested_model: Annotated[NestedModel, Field()]):
        pass

    schema = jsonschema_from_task_func(f)
    nested_model_def = schema["$defs"]["NestedModel"]
    assert set(nested_model_def["properties"]) == {"basic_field", "advanced_field"}
    assert set(nested_model_def["required"]) == {"basic_field"}


def test_advanced_field_nested_dataclass_annotated():
    @dataclass
    class NestedModel:
        basic_field: Annotated[str, Field()]
        advanced_field: Annotated[str, AdvancedField(default=None)]

    def f(nested_model: Annotated[NestedModel, Field()]):
        pass

    schema = jsonschema_from_task_func(f)
    nested_model_def = schema["$defs"]["NestedModel"]
    assert set(nested_model_def["properties"]) == {"basic_field", "advanced_field"}
    assert set(nested_model_def["required"]) == {"basic_field"}


def test_advanced_field_nested_dataclass_not_annotated():
    @dataclass
    class NestedModel:
        basic_field: str = Field()
        advanced_field: str = AdvancedField(default=None)

    def f(nested_model: Annotated[NestedModel, Field()]):
        pass

    schema = jsonschema_from_task_func(f)
    nested_model_def = schema["$defs"]["NestedModel"]
    assert set(nested_model_def["properties"]) == {"basic_field", "advanced_field"}
    assert set(nested_model_def["required"]) == {"basic_field"}


def test_advanced_field_task_annotation_without_default_raises():
    expected_error_message = "A default is required for fields constructed with 'AdvancedField'."

    with pytest.raises(ValueError, match=expected_error_message):
        AdvancedField()

    with pytest.raises(ValueError, match=expected_error_message):

        def f(advanced_field: Annotated[str, AdvancedField()]):
            pass


def test_advanced_field_default_can_be_arg_or_kw():
    does_not_raise = AdvancedField(1)
    assert does_not_raise.default == 1

    also_does_not_raise = AdvancedField(default=1)
    assert also_does_not_raise.default == 1


def test_advanced_field_json_schema_extra():
    def f(
        basic_field: Annotated[str, Field()],
        advanced_field: Annotated[str, AdvancedField(default=None)],
        with_extra: Annotated[
            str,
            AdvancedField(default=None, json_schema_extra={"arbitrary:field": True}),
        ],
    ):
        pass

    schema = jsonschema_from_task_func(f)
    assert set(schema["properties"]) == {"basic_field", "advanced_field", "with_extra"}
    assert set(schema["required"]) == {"basic_field"}
    assert schema["properties"]["advanced_field"]["ecoscope:advanced"]
    assert schema["properties"]["with_extra"]["ecoscope:advanced"]
    assert schema["properties"]["with_extra"]["arbitrary:field"]


def test_advanced_field_json_schema_extra_raises_on_conflict():
    expected_error_message = (
        "Fields constructed with 'AdvancedField' cannot override "
        r"json_schema_extra: dict_keys\(\['ecoscope:advanced'\]\)"
    )
    with pytest.raises(ValueError, match=expected_error_message):
        AdvancedField(default=None, json_schema_extra={"ecoscope:advanced": False})


def test_dataframe_type():
    df = pd.DataFrame(data={"col1": [1, 2]})

    class ValidSchema(JsonSerializableDataFrameModel):
        col1: pa_typing.Series[int] = pa.Field(unique=True)

    class ValidModel(BaseModel):
        df: DataFrame[ValidSchema]

    ValidModel(df=df)

    class InvalidSchema(JsonSerializableDataFrameModel):
        # Invalid because col1 elements are ints, not strings
        col1: pa_typing.Series[str] = pa.Field(unique=True)

    class InvalidModel(BaseModel):
        df: DataFrame[InvalidSchema]

    with pytest.raises(ValidationError):
        InvalidModel(df=df)

    class EmptyModel(BaseModel):
        df: DataFrame[EmptyDataFrameModel]

    EmptyModel(df=pd.DataFrame())

    with pytest.raises(ValidationError):
        EmptyModel(df=df)
