import pytest

from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema


class MatchingSchema(GenerateJsonSchema):
    def generate(self, schema, mode='validation'):
        json_schema = super().generate(schema, mode=mode)
        if "title" in json_schema:
            # TypeAdapter(func).json_schema() has no "title"
            del json_schema["title"]
        # By default, this is False for TypeAdapter(func).json_schema(),
        # and omitted for BaseModel.model_json_schema(). Another way to
        # get this to match is to use ConfigDict(extra=False) as config
        # for the BaseModel, since we're subclassing GenerateJsonSchema
        # anyway, this works too...
        json_schema["additionalProperties"] = False
        return json_schema


def test_jsonschema_from_signature_basic():
    class FuncSignature(BaseModel):
        foo: int
        bar: str

    def func(foo: int, bar: str): ...

    from_func = TypeAdapter(func).json_schema(schema_generator=MatchingSchema)
    from_model = FuncSignature.model_json_schema(schema_generator=MatchingSchema)
    assert from_func == from_model
