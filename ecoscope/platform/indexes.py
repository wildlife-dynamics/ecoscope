"""Grouper types and temporal/spatial index definitions.

Defines temporal index dataclasses (Year, Month, Date, Hour, etc.), spatial and
value-based groupers, and the ``UserDefinedGroupers`` type alias used to
configure how workflow data is split into groups.
"""

import calendar
from dataclasses import dataclass
from typing import Annotated, Callable, List, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, PrivateAttr
from pydantic.functional_validators import BeforeValidator
from pydantic.json_schema import JsonSchemaValue, SkipJsonSchema
from pydantic_core import core_schema as cs

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.jsonschema import oneOf

IndexName: TypeAlias = str
IndexValue: TypeAlias = str
Filter = tuple[IndexName, Literal["="], IndexValue]
CompositeFilter = tuple[Filter, ...]


@dataclass(frozen=True)
class Year:
    directive: Literal["%Y"] = "%Y"
    selector_title: str = "Year (example: 2024)"
    display_name: str = "Year"
    sort_key: None = None


@dataclass(frozen=True)
class Month:
    directive: Literal["%B"] = "%B"
    selector_title: str = "Month (example: September)"
    display_name: str = "Month"
    sort_key: Callable = list(calendar.month_name).index


@dataclass(frozen=True)
class YearMonth:
    directive: Literal["%Y-%m"] = "%Y-%m"
    selector_title: str = "Year and Month (example: 2023-01)"
    display_name: str = "Year-Month"
    sort_key: None = None


@dataclass(frozen=True)
class DayOfTheYear:
    directive: Literal["%j"] = "%j"
    selector_title: str = "Day of the year as a number (example: 365)"
    display_name: str = "Day of the year"
    sort_key: None = None


@dataclass(frozen=True)
class DayOfTheMonth:
    directive: Literal["%d"] = "%d"
    selector_title: str = "Day of the month as a number (example: 31)"
    display_name: str = "Day of the month"
    sort_key: None = None


@dataclass(frozen=True)
class Date:
    directive: Literal["%Y-%m-%d"] = "%Y-%m-%d"
    selector_title: str = "Date (example: 2025-01-31)"
    display_name: str = "Date"
    sort_key: None = None


@dataclass(frozen=True)
class DayOfTheWeek:
    directive: Literal["%A"] = "%A"
    selector_title: str = "Day of the week (example: Sunday)"
    display_name: str = "Day of the week"
    sort_key: Callable = list(calendar.day_name).index


@dataclass(frozen=True)
class Hour:
    directive: Literal["%H"] = "%H"
    selector_title: str = "Hour (24-hour clock) as number (example: 22)"
    display_name: str = "Hour"
    sort_key: None = None


TemporalIndexType = Union[
    Year,
    Month,
    YearMonth,
    DayOfTheYear,
    DayOfTheMonth,
    DayOfTheWeek,
    Hour,
    Date,
]


strftime_directives: List[TemporalIndexType] = [
    Year(),
    Month(),
    YearMonth(),
    DayOfTheYear(),
    DayOfTheMonth(),
    DayOfTheWeek(),
    Hour(),
    Date(),
]


def _coerce_temporal_index(v):
    """Convert a strftime directive string to the corresponding TemporalIndexType instance."""
    if isinstance(v, str):
        for t in strftime_directives:
            if t.directive == v:
                return t
        raise ValueError(f"Unknown temporal index directive: {v}")
    return v


class AllGrouper(BaseModel):
    model_config = ConfigDict(frozen=True)

    index_name: Annotated[str, AdvancedField(default="All")] = "All"

    @property
    def display_name(self):
        return self.index_name.title().replace("_", " ")

    @property
    def help_text(self):
        # NOTE(cisaacstern): For compatibility with the grouper interface, including this
        # as a property. But making it read-only so that it doesn't show up in RJSF-land.
        return None

    @property
    def sort_key(self) -> None:
        return None


class ValueGrouper(BaseModel):
    model_config = ConfigDict(frozen=True, title="Category")

    index_name: IndexName
    help_text: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None, exclude=True)] = None

    @property
    def display_name(self):
        return self.index_name.replace("_", " ").title()

    @property
    def sort_key(self) -> None:
        return None

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Override the json schema generation of this object for the RJSF UI."""
        return {
            "type": "object",
            "title": "Category",
            "properties": {
                "index_name": {
                    "oneOf": [],  # to be overridden in spec.yaml `rjsf-overrides` section
                    "title": "Category",
                    "type": "string",
                },
            },
            "required": ["index_name"],
        }


class TemporalGrouper(BaseModel):
    model_config = ConfigDict(frozen=True, title="Time")

    temporal_index: Annotated[TemporalIndexType, BeforeValidator(_coerce_temporal_index)]
    help_text: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None, exclude=True)] = None
    is_temporal: Annotated[Literal[True], Field(exclude=True)] = True

    @property
    def index_name(self):
        return f"TemporalGrouper_{self.temporal_index.directive}"

    @property
    def display_name(self):
        return self.temporal_index.display_name

    @property
    def sort_key(self) -> Callable | None:
        return self.temporal_index.sort_key

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Override the json schema generation of this object for the RJSF UI."""
        return {
            "type": "object",
            "title": "Time",
            "properties": {
                "temporal_index": {
                    "oneOf": [
                        oneOf(const=d.directive, title=d.selector_title).model_dump() for d in strftime_directives
                    ],
                    "title": "Time",
                    "type": "string",
                },
            },
            "required": ["temporal_index"],
        }


class SpatialGrouper(BaseModel):
    model_config = ConfigDict(title="Spatial")

    spatial_index_name: IndexName
    help_text: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None, exclude=True)] = None
    _resolved_spatial_regions: AnyGeoDataFrame | None = PrivateAttr(default=None)

    @property
    def index_name(self) -> str:
        return f"SpatialGrouper_{self.spatial_index_name}"

    @property
    def sort_key(self) -> None:
        return None

    @property
    def display_name(self) -> str:
        return self.spatial_index_name

    @property
    def spatial_regions(self) -> AnyGeoDataFrame | None:
        return self._resolved_spatial_regions

    @property
    def is_resolved(self) -> bool:
        return self._resolved_spatial_regions is not None

    def __hash__(self) -> int:
        return hash(("SpatialGrouper", self.spatial_index_name))

    def __eq__(self, other) -> bool:
        if isinstance(other, SpatialGrouper):
            return self.spatial_index_name == other.spatial_index_name
        return False

    def resolve(self, spatial_regions: AnyGeoDataFrame) -> None:
        self._resolved_spatial_regions = spatial_regions

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Override the json schema generation of this object for the RJSF UI."""
        return {
            "type": "object",
            "title": "Spatial",
            "properties": {
                "spatial_index_name": {"title": "Spatial Regions", "type": "string"},
            },
            "required": ["spatial_index_name"],
        }


UserDefinedGroupers: TypeAlias = list[ValueGrouper | TemporalGrouper | SpatialGrouper]
