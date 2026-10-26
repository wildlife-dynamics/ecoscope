import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import ismethod
from typing import Annotated, Any, ClassVar, Generic, Protocol, Type, TypeVar, get_args, runtime_checkable

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic.functional_validators import BeforeValidator
from pydantic.json_schema import WithJsonSchema
from pydantic_settings import BaseSettings, SettingsConfigDict

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame

logger = logging.getLogger(__name__)

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
    server: str
    token: str | None

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
        patrols_overlap_daterange,
    ) -> AnyGeoDataFrame: ...

    def get_patrol_events(
        self,
        since,
        until,
        patrol_type_value,
        event_type,
        status,
        event_state,
        drop_null_geometry,
        sub_page_size,
        patrols_overlap_daterange,
    ) -> AnyGeoDataFrame: ...

    def get_events(
        self,
        since,
        until,
        event_type,
        state,
        drop_null_geometry,
        include_details: bool,
        include_updates: bool,
        include_related_events: bool,
        force_point_geometry: bool,
    ) -> AnyGeoDataFrame: ...

    def get_event_types(self) -> AnyDataFrame: ...

    def get_patrols(
        self,
        since,
        until,
        patrol_type_value,
        status,
        sub_page_size,
        patrols_overlap_daterange,
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


@dataclass
class EarthRangerBackends:
    """The backends resolved for one named EarthRanger data source.

    ``warehouse`` is None when this source reads the EarthRanger API, so per-source
    eligibility (ERA-855) is expressed by what the connection built rather than by a
    flag the task layer has to interpret. Routing stays with the tasks: an observation
    read takes the warehouse whenever one is here, while event reads also consult the
    ``DWH_EVENTS_ENABLED`` kill switch.

    Tasks annotate the `EarthRangerSource` alias to be handed one of these. The older
    `EarthRangerClient` alias still resolves to the API client alone, which is what the
    ext task libraries call directly (`client._get`, `client.tcp_limit`, ...). Those
    libraries also call built-in tasks directly with that client, so the tasks pass it
    through `as_backends`.
    """

    api: EarthRangerClientProtocol
    # Any, not ERWarehouseClient: pydantic builds a schema for this dataclass and so
    # resolves the annotation at runtime, which rules out a TYPE_CHECKING-only import,
    # and importing io-core eagerly here would put it on the import path of everything
    # that touches connections. Typing it buys nothing anyway -- io-core ships no
    # py.typed, so mypy reads ERWarehouseClient as Any regardless.
    warehouse: Any | None = None


def _make_warehouse_client(server: str, token: SecretStr) -> Any:
    from ecoscope_earthranger_io_core.client import ERWarehouseClient  # type: ignore[import-untyped]

    warehouse_base_url = os.environ.get("EARTHRANGER_WAREHOUSE_API_BASE_URL")
    logger.debug("Using ERWarehouseClient (warehouse_base_url=%s)", warehouse_base_url)
    return ERWarehouseClient(server=server, token=token, warehouse_base_url=warehouse_base_url)


def as_backends(client: "EarthRangerBackends | EarthRangerClientProtocol") -> EarthRangerBackends:
    """Accept a bare API client where tasks expect `EarthRangerBackends`.

    The `EarthRangerSource` validator only runs when a task's arguments are validated,
    so a direct call from code holding an `EarthRangerClient` arrives here unresolved.
    Such a client carries no per-source preference, so it gets the global flag's
    warehouse, which is how these tasks behaved before `EarthRangerSource`.
    """
    if isinstance(client, EarthRangerBackends):
        return client
    warnings.warn(
        "Passing an EarthRanger API client to a task that takes `EarthRangerSource` is "
        "deprecated and ignores the per-source `use_warehouse` setting; pass "
        "`EarthRangerSourceConnection.client_from_named_connection(name)` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    token = getattr(client, "token", None)
    warehouse = (
        _make_warehouse_client(client.server, SecretStr(token))  # type: ignore[attr-defined]
        if _warehouse_enabled_globally() and token
        else None
    )
    return EarthRangerBackends(api=client, warehouse=warehouse)


def _warehouse_enabled_globally() -> bool:
    """The legacy all-or-nothing warehouse switch.

    Now a fallback: `EarthRangerConnection.use_warehouse` overrides it per data source,
    so this decides only for sources that express no preference -- which is every source
    on an older ecoscope-server, since that sends this flag alone.
    """
    return os.environ.get("USE_EARTHRANGER_WAREHOUSE_API", "false").lower() == "true"


class _EarthRangerConnection(_DataConnection):
    """Everything about a named EarthRanger connection except what it resolves to.

    Two concrete connections share it so each can declare its own client type: tasks
    take either the API client alone (`EarthRangerClient`) or both backends
    (`EarthRangerSource`), and `DataConnection` is generic in exactly that.
    """

    __ecoscope_connection_type__: ClassVar[str] = "earthranger"

    server: Annotated[str, Field(description="EarthRanger API URL")]
    username: Annotated[str, Field(description="EarthRanger username")] = ""
    password: Annotated[SecretStr, Field(description="EarthRanger password")] = SecretStr("")
    tcp_limit: Annotated[int, Field(description="TCP limit for API requests")] = 5
    sub_page_size: Annotated[int, Field(description="Sub page size for API requests")] = 4000
    token: Annotated[SecretStr, Field(description="EarthRanger access token")] = SecretStr("")
    use_warehouse: Annotated[
        bool | None,
        Field(
            description=(
                "Read this data source through the EarthRanger data warehouse. "
                "Unset defers to the global USE_EARTHRANGER_WAREHOUSE_API flag."
            )
        ),
    ] = None

    @field_validator("use_warehouse", mode="before")
    @classmethod
    def blank_use_warehouse_is_unset(cls, v: Any) -> Any:
        # pydantic-settings hands an empty env var through as "", which `bool | None`
        # rejects -- failing the whole connection, even for tasks that never touch the
        # warehouse. The legacy global flag read blank as "no preference"; so does this.
        if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
            return None
        return v

    @field_validator("token")
    def token_or_password(cls, v: SecretStr, info: ValidationInfo):
        if v and (info.data["username"] or info.data["password"]):
            raise ValueError("Only a token, or an EarthRanger username and password can be provided, not both")
        if not v and not (info.data["username"] and info.data["password"]):
            raise ValueError("If token is empty, EarthRanger username and password must be provided")
        return v

    def warehouse_enabled(self) -> bool:
        """Whether this data source reads through the EarthRanger data warehouse.

        The per-source opt-in wins; unset defers to the global flag (ERA-855).
        """
        return self.use_warehouse if self.use_warehouse is not None else _warehouse_enabled_globally()

    def _warehouse_client(self) -> Any | None:
        """This source's warehouse backend, or None when it reads the EarthRanger API.

        Token-only: the warehouse authenticates with the EarthRanger token, so a
        username/password connection has nothing to present and stays on the API.
        """
        if not self.warehouse_enabled() or not self.token:
            return None
        return _make_warehouse_client(self.server, self.token)

    def _api_client(self) -> EarthRangerClientProtocol:
        from ecoscope.io import EarthRangerIO

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


class EarthRangerConnection(_EarthRangerConnection, DataConnection[EarthRangerClientProtocol]):
    """Resolves a named EarthRanger connection to its API client."""

    def get_client(self) -> EarthRangerClientProtocol:
        return self._api_client()


class EarthRangerSourceConnection(_EarthRangerConnection, DataConnection[EarthRangerBackends]):
    """Resolves a named EarthRanger connection to both of its backends.

    A separate connection rather than a flag on `EarthRangerConnection` because
    `is_client` keys off the `client_from_named_connection` validator, so each
    annotation needs its own binding. The connection type -- and so the env prefix --
    is the same, which is what lets one configured source serve both.
    """

    def get_client(self) -> EarthRangerBackends:
        return EarthRangerBackends(api=self._api_client(), warehouse=self._warehouse_client())


@runtime_checkable
class SmartClientProtocol(Protocol):
    def get_patrol_observations(
        self,
        ca_uuid: str,
        language_uuid: str,
        start: str,
        end: str,
        patrol_mandate: str | None = ...,
        patrol_transport: str | None = ...,
        station_uuid: str | None = ...,
    ) -> Any: ...

    def get_events(self, ca_uuid: str, language_uuid: str, start: str, end: str) -> AnyGeoDataFrame: ...


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
        from ecoscope.io import SmartIO

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
        from ecoscope.io import EarthEngineIO

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

EarthRangerSource = Annotated[
    EarthRangerBackends,
    BeforeValidator(EarthRangerSourceConnection.client_from_named_connection),
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
