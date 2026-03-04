from abc import ABC, abstractmethod
from inspect import ismethod
from typing import Annotated, ClassVar, Generic, Protocol, Type, TypeVar, get_args, runtime_checkable

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic.functional_validators import BeforeValidator
from pydantic.json_schema import WithJsonSchema
from pydantic_settings import BaseSettings, SettingsConfigDict

DataConnectionType = TypeVar("DataConnectionType", bound="DataConnection")
ClientProtocolType = TypeVar("ClientProtocolType")


class _DataConnection(BaseSettings):
    @classmethod
    def from_named_connection(  # type: ignore[misc]
        cls: Type[DataConnectionType], name: str
    ) -> DataConnectionType:
        model_config = SettingsConfigDict(
            env_prefix=(f"ecoscope_workflows__connections__{cls.__ecoscope_connection_type__}__{name.lower()}__"),
            case_sensitive=False,
            pyproject_toml_table_header=(
                "connections",
                cls.__ecoscope_connection_type__,
                name,
            ),
        )
        _cls = type(
            f"{name}_connection",
            (cls,),
            {"model_config": model_config},
        )
        return _cls()


class DataConnection(ABC, _DataConnection, Generic[ClientProtocolType]):
    __ecoscope_connection_type__: ClassVar[str] = NotImplemented

    @abstractmethod
    def get_client(self) -> ClientProtocolType: ...

    # @abstractmethod
    # def check_connection(self) -> None: ...

    @classmethod
    def client_from_named_connection(cls, name: str) -> ClientProtocolType:
        return cls.from_named_connection(name).get_client()


def is_client(obj):
    if hasattr(obj, "__origin__") and hasattr(obj, "__args__"):
        if any(isinstance(arg, BeforeValidator) for arg in get_args(obj)):
            bv = [arg for arg in get_args(obj) if isinstance(arg, BeforeValidator)][0]
            if ismethod(bv.func) and bv.func.__name__ == "client_from_named_connection":
                return True
    return False


def connection_from_client(obj) -> DataConnection:
    assert is_client(obj)
    bv = [arg for arg in get_args(obj) if isinstance(arg, BeforeValidator)][0]
    conn_type = bv.func.__self__  # type: ignore[union-attr]
    assert issubclass(conn_type, DataConnection)
    return conn_type


@runtime_checkable
class EarthRangerClientProtocol(Protocol):
    def get_subjectgroup_observations(
        self,
        subject_group_name: str,
        include_subject_details: bool,
        include_inactive: bool,
        include_details: bool,
        include_subjectsource_details: bool,
        since,
        until,
        filter,
    ) -> AnyGeoDataFrame: ...

    def get_patrol_observations_with_patrol_filter(
        self,
        since,
        until,
        patrol_type_value,
        status,
        include_patrol_details,
        sub_page_size,
    ) -> AnyGeoDataFrame: ...

    def get_patrol_events(
        self,
        since,
        until,
        patrol_type_value,
        event_type,
        status,
        drop_null_geometry,
        sub_page_size,
    ) -> AnyGeoDataFrame: ...

    def get_events(
        self,
        since,
        until,
        event_type,
        drop_null_geometry,
        include_details: bool,
        include_updates: bool,
        include_related_events: bool,
    ) -> AnyGeoDataFrame: ...

    def get_event_types(self) -> AnyDataFrame: ...

    def get_patrols(
        self,
        since,
        until,
        patrol_type_value,
        status,
        sub_page_size,
    ) -> AnyDataFrame: ...

    def get_patrol_observations(
        self,
        patrols_df,
        include_patrol_details,
        sub_page_size,
    ) -> AnyGeoDataFrame: ...

    def get_event_type_display_names_from_events(
        self,
        events_gdf,
        append_category_names,
    ) -> AnyGeoDataFrame: ...

    def get_choices_from_v2_event_type(self, event_type, choice_field) -> dict[str, str]: ...

    def get_spatial_features_group(
        self,
        spatial_features_group_name,
        spatial_features_group_id,
        with_group_data,
    ) -> dict[str, str | int | AnyGeoDataFrame]: ...

    def get_fields_from_event_type_schema(self, event_type) -> dict[str, str]: ...


class EarthRangerConnection(DataConnection[EarthRangerClientProtocol]):
    __ecoscope_connection_type__: ClassVar[str] = "earthranger"

    server: Annotated[str, Field(description="EarthRanger API URL")]
    username: Annotated[str, Field(description="EarthRanger username")] = ""
    password: Annotated[SecretStr, Field(description="EarthRanger password")] = SecretStr("")
    tcp_limit: Annotated[int, Field(description="TCP limit for API requests")] = 5
    sub_page_size: Annotated[int, Field(description="Sub page size for API requests")] = 4000
    token: Annotated[SecretStr, Field(description="EarthRanger access token")] = SecretStr("")

    @field_validator("token")
    def token_or_password(cls, v: SecretStr, info: ValidationInfo):
        if v and (info.data["username"] or info.data["password"]):
            raise ValueError("Only a token, or an EarthRanger username and password can be provided, not both")
        if not v and not (info.data["username"] and info.data["password"]):
            raise ValueError("If token is empty, EarthRanger username and password must be provided")
        return v

    def get_client(self) -> EarthRangerClientProtocol:
        from ecoscope.io import EarthRangerIO  # type: ignore[import-untyped]

        auth_kws = (
            {"token": self.token.get_secret_value()}
            if self.token
            else {
                "username": self.username,
                "password": self.password.get_secret_value(),
            }
        )
        return EarthRangerIO(
            server=self.server,
            tcp_limit=self.tcp_limit,
            sub_page_size=self.sub_page_size,
            **auth_kws,
        )


@runtime_checkable
class SmartClientProtocol(Protocol):
    def get_patrol_observations(
        self,
        ca_uuid,
        language_uuid,
        start,
        end,
        patrol_mandate,
        patrol_transport,
    ) -> AnyGeoDataFrame: ...

    def get_events(self, ca_uuid, language_uuid, start, end) -> AnyGeoDataFrame: ...


class SmartConnection(DataConnection[SmartClientProtocol]):
    __ecoscope_connection_type__: ClassVar[str] = "smart"

    server: Annotated[str, Field(description="Smart API URL")]
    username: Annotated[str, Field(description="Smart username")] = ""
    password: Annotated[SecretStr, Field(description="Smart password")] = SecretStr("")
    token: Annotated[SecretStr, Field(description="Smart access token")] = SecretStr("")

    @field_validator("token")
    def token_or_password(cls, v: SecretStr, info: ValidationInfo):
        if v and (info.data["username"] or info.data["password"]):
            raise ValueError("Only a token, or an EarthRanger username and password can be provided, not both")
        if not v and not (info.data["username"] and info.data["password"]):
            raise ValueError("If token is empty, SMART username and password must be provided")
        return v

    def get_client(self) -> SmartClientProtocol:
        from ecoscope.io import SmartIO  # type: ignore[import-untyped]

        auth_kws = (
            {"token": self.token.get_secret_value()}
            if self.token
            else {
                "username": self.username,
                "password": self.password.get_secret_value(),
            }
        )
        return SmartIO(
            urlBase=self.server,
            **auth_kws,
        )


@runtime_checkable
class EarthEngineClientProtocol(Protocol):
    pass


class EarthEngineConnection(DataConnection[EarthEngineClientProtocol]):
    __ecoscope_connection_type__: ClassVar[str] = "earthengine"

    service_account: Annotated[str, Field(description="Your Google Cloud Service Account")] = ""
    private_key: Annotated[SecretStr, Field(description="Your service account's private key")] = SecretStr("")
    private_key_file: Annotated[str, Field(description="Your service account's private key")] = ""
    ee_project: Annotated[str, Field(description="Your EarthEngine project ID")] = ""

    def get_client(self):
        from ecoscope.io import EarthEngineIO  # type: ignore[import-untyped]

        return EarthEngineIO(
            service_account=self.service_account,
            private_key=self.private_key.get_secret_value(),
            private_key_file=self.private_key_file,
            ee_project=self.ee_project,
        )


EarthRangerClient = Annotated[
    EarthRangerClientProtocol,
    BeforeValidator(EarthRangerConnection.client_from_named_connection),
    WithJsonSchema({"type": "string", "description": "A named EarthRanger connection."}),
]

SmartClient = Annotated[
    SmartClientProtocol,
    BeforeValidator(SmartConnection.client_from_named_connection),
    WithJsonSchema({"type": "string", "description": "A named SMART connection."}),
]

EarthEngineClient = Annotated[
    EarthEngineClientProtocol,
    BeforeValidator(EarthEngineConnection.client_from_named_connection),
    WithJsonSchema({"type": "string", "description": "A named Google EarthEngine connection."}),
]
