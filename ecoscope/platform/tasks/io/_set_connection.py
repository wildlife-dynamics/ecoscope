from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField

ConnectionName = Annotated[str, Field(title="Data Source")]
DataSourceField = Field(
    title="",
    description="Select one of your configured data sources.",
)


@dataclass(frozen=True)
class EarthRangerConnection:
    name: ConnectionName
    connection_type: Annotated[Literal["EarthRanger"], AdvancedField(default="EarthRanger", exclude=True)] = (
        "EarthRanger"
    )


@dataclass(frozen=True)
class SMARTConnection:
    name: ConnectionName
    connection_type: Annotated[Literal["SMART"], AdvancedField(default="SMART", exclude=True)] = "SMART"


@dataclass(frozen=True)
class GoogleEarthEngineConnection:
    name: ConnectionName
    connection_type: Annotated[
        Literal["GoogleEarthEngine"],
        AdvancedField(default="GoogleEarthEngine", exclude=True),
    ] = "GoogleEarthEngine"


@register()
def set_er_connection(
    data_source: Annotated[
        EarthRangerConnection,
        DataSourceField,
    ],
) -> str:
    return data_source.name


@register()
def set_smart_connection(
    data_source: Annotated[
        SMARTConnection,
        DataSourceField,
    ],
) -> str:
    return data_source.name


@register()
def set_gee_connection(
    data_source: Annotated[
        GoogleEarthEngineConnection,
        DataSourceField,
    ],
) -> str:
    return data_source.name
