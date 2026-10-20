import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, validate_call

from ecoscope.platform.annotations import DataFrame
from ecoscope.platform.connections import (
    EarthEngineClient,
    EarthEngineConnection,
    EarthRangerBackends,
    EarthRangerClient,
    EarthRangerConnection,
    EarthRangerSource,
    EarthRangerSourceConnection,
    SmartClient,
    SmartConnection,
    is_client,
)


def test_is_client():
    assert is_client(EarthRangerClient)
    assert is_client(EarthRangerSource)
    assert is_client(SmartClient)
    assert is_client(EarthEngineClient)
    assert not is_client(DataFrame)


def test_er_connection_unnamed():
    mock_env = {
        "SERVER": "https://earthranger.com",
        "USERNAME": "user",
        "PASSWORD": "pass",
        "TCP_LIMIT": "5",
        "SUB_PAGE_SIZE": "4000",
    }
    with patch.dict(os.environ, mock_env):
        conn = EarthRangerConnection()
        assert conn.server == "https://earthranger.com"
        assert conn.username == "user"

        assert isinstance(conn.password, SecretStr)
        assert str(conn.password) == "**********"
        assert conn.password.get_secret_value() == "pass"

        assert conn.tcp_limit == 5
        assert conn.sub_page_size == 4000


def test_smart_connection_unnamed():
    mock_env = {
        "SERVER": "https://smart.com",
        "USERNAME": "user",
        "PASSWORD": "pass",
    }
    with patch.dict(os.environ, mock_env):
        conn = SmartConnection()
        assert conn.server == "https://smart.com"
        assert conn.username == "user"

        assert isinstance(conn.password, SecretStr)
        assert str(conn.password) == "**********"
        assert conn.password.get_secret_value() == "pass"


def test_ee_connection_unnamed():
    mock_env = {
        "SERVICE_ACCOUNT": "service",
        "PRIVATE_KEY": "key",
    }
    with patch.dict(os.environ, mock_env):
        conn = EarthEngineConnection()
        assert conn.service_account == "service"

        assert isinstance(conn.private_key, SecretStr)
        assert str(conn.private_key) == "**********"
        assert conn.private_key.get_secret_value() == "key"


@pytest.fixture
def named_mock_env():
    return {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__SERVER": ("https://mep-dev.pamdas.org"),
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USERNAME": "user",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__PASSWORD": "pass",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__TCP_LIMIT": "5",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__SUB_PAGE_SIZE": "4000",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__SMART__TEST__SERVER": (
            "https://smartapitest.smartconservationtools.org/smartapi/"
        ),
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__SMART__TEST__USERNAME": "user",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__SMART__TEST__PASSWORD": "pass",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHENGINE__ECOSCOPE_POC__SERVICE_ACCOUNT": "service",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHENGINE__ECOSCOPE_POC__PRIVATE_KEY": "key",
    }


def test_er_connection_named_from_env(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        conn = EarthRangerConnection.from_named_connection("MEP_DEV")
        assert conn.server == "https://mep-dev.pamdas.org"
        assert conn.username == "user"

        assert isinstance(conn.password, SecretStr)
        assert str(conn.password) == "**********"
        assert conn.password.get_secret_value() == "pass"

        assert conn.tcp_limit == 5
        assert conn.sub_page_size == 4000


def test_smart_connection_named_from_env(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        conn = SmartConnection.from_named_connection("TEST")
        assert conn.server == "https://smartapitest.smartconservationtools.org/smartapi/"
        assert conn.username == "user"

        assert isinstance(conn.password, SecretStr)
        assert str(conn.password) == "**********"
        assert conn.password.get_secret_value() == "pass"


def test_ee_connection_named_from_env(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        conn = EarthEngineConnection.from_named_connection("ecoscope_poc")
        assert conn.service_account == "service"

        assert isinstance(conn.private_key, SecretStr)
        assert str(conn.private_key) == "**********"
        assert conn.private_key.get_secret_value() == "key"


def test_resolve_er_client_from_env(named_mock_env):
    @validate_call(config={"arbitrary_types_allowed": True})
    def f(client: EarthRangerClient):
        return client

    with patch.dict(os.environ, named_mock_env):
        with patch("ecoscope.io.EarthRangerIO", autospec=True) as mock_cls:
            mock_cls.return_value.server = "https://mep-dev.pamdas.org"
            mock_cls.return_value.token = None
            client = f(client="mep_dev")
            assert callable(client.get_subjectgroup_observations)


@pytest.fixture
def named_mock_env_with_token():
    return {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__SERVER": ("https://mep-dev.pamdas.org"),
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__TOKEN": "123456789",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__TCP_LIMIT": "5",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__SUB_PAGE_SIZE": "4000",
    }


def test_connection_named_from_env_with_token(named_mock_env_with_token):
    with patch.dict(os.environ, named_mock_env_with_token):
        conn = EarthRangerConnection.from_named_connection("MEP_DEV")
        assert conn.server == "https://mep-dev.pamdas.org"

        assert isinstance(conn.token, SecretStr)
        assert str(conn.token) == "**********"
        assert conn.token.get_secret_value() == "123456789"

        assert conn.tcp_limit == 5
        assert conn.sub_page_size == 4000


@pytest.mark.parametrize(
    "env_value,expected",
    [("true", True), ("True", True), ("1", True), ("false", False), ("False", False), ("0", False)],
)
def test_er_connection_use_warehouse_from_env(named_mock_env_with_token, env_value, expected):
    env = named_mock_env_with_token | {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": env_value
    }
    with patch.dict(os.environ, env):
        conn = EarthRangerConnection.from_named_connection("MEP_DEV")
        assert conn.use_warehouse is expected


def test_er_connection_use_warehouse_defaults_to_none(named_mock_env_with_token):
    """Unset must stay distinct from False: it defers to the global warehouse flag."""
    with patch.dict(os.environ, named_mock_env_with_token, clear=True):
        conn = EarthRangerConnection.from_named_connection("MEP_DEV")
        assert conn.use_warehouse is None


def test_er_connection_use_warehouse_is_per_connection(named_mock_env_with_token):
    env = named_mock_env_with_token | {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": "true",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__OTHER__SERVER": "https://other.pamdas.org",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__OTHER__TOKEN": "987654321",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__OTHER__USE_WAREHOUSE": "false",
    }
    with patch.dict(os.environ, env, clear=True):
        assert EarthRangerConnection.from_named_connection("MEP_DEV").use_warehouse is True
        assert EarthRangerConnection.from_named_connection("OTHER").use_warehouse is False


def test_er_client_annotation_still_resolves_to_the_api_client(named_mock_env_with_token):
    """The ext task libraries call the API client directly off this annotation
    (`client._get`, `client.tcp_limit`, ...), so it must not become a source."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def f(client: EarthRangerClient):
        return client

    with patch.dict(os.environ, named_mock_env_with_token, clear=True):
        with patch("ecoscope.io.EarthRangerIO", autospec=True) as mock_cls:
            # server/token are protocol members, and autospec does not create
            # annotation-only attributes, so the isinstance check needs them set.
            mock_cls.return_value.server = "https://mep-dev.pamdas.org"
            mock_cls.return_value.token = "123456789"
            mock_cls.return_value._get.return_value = {"results": []}
            client = f(client="mep_dev")

    assert not hasattr(client, "warehouse")
    assert client._get("/activity/event/evt-1/files") == {"results": []}


def test_er_source_annotation_resolves_to_both_backends(named_mock_env_with_token):
    with patch.dict(os.environ, named_mock_env_with_token, clear=True):
        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            source = EarthRangerSourceConnection.from_named_connection("MEP_DEV").get_client()

    assert isinstance(source, EarthRangerBackends)
    assert source.warehouse is None
    assert callable(source.api.get_subjectgroup_observations)


def test_er_connection_get_client_builds_a_warehouse_backend(named_mock_env_with_token):
    from ecoscope_earthranger_io_core.client import ERWarehouseClient

    env = named_mock_env_with_token | {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": "true",
        "EARTHRANGER_WAREHOUSE_API_BASE_URL": "http://warehouse-test",
    }
    with patch.dict(os.environ, env, clear=True):
        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            source = EarthRangerSourceConnection.from_named_connection("MEP_DEV").get_client()

    assert isinstance(source.warehouse, ERWarehouseClient)
    assert source.warehouse.warehouse_base_url == "http://warehouse-test"


def test_er_connection_warehouse_backend_without_a_base_url(named_mock_env_with_token):
    """EARTHRANGER_WAREHOUSE_API_BASE_URL is optional; io-core falls back to its default."""
    from ecoscope_earthranger_io_core.client import ERWarehouseClient

    env = named_mock_env_with_token | {"ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": "true"}
    with patch.dict(os.environ, env, clear=True):
        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            source = EarthRangerSourceConnection.from_named_connection("MEP_DEV").get_client()

    assert isinstance(source.warehouse, ERWarehouseClient)
    assert source.warehouse.warehouse_base_url is None


def test_er_connection_get_client_has_no_warehouse_backend_when_opted_out(named_mock_env_with_token):
    env = named_mock_env_with_token | {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": "false",
        "USE_EARTHRANGER_WAREHOUSE_API": "true",
    }
    with patch.dict(os.environ, env, clear=True):
        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            source = EarthRangerSourceConnection.from_named_connection("MEP_DEV").get_client()

    assert source.warehouse is None


def test_er_connection_without_a_token_has_no_warehouse_backend(named_mock_env):
    """The warehouse authenticates with the ER token, so username/password stays on the API."""
    env = named_mock_env | {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": "true",
    }
    with patch.dict(os.environ, env, clear=True):
        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            source = EarthRangerSourceConnection.from_named_connection("MEP_DEV").get_client()

    assert source.warehouse is None


def test_two_sources_resolve_their_backends_independently(named_mock_env_with_token):
    """One workflow, two ER sources, different backends (ERA-855)."""
    from ecoscope_earthranger_io_core.client import ERWarehouseClient

    env = named_mock_env_with_token | {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": "true",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__OTHER__SERVER": "https://other.pamdas.org",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__OTHER__TOKEN": "987654321",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__OTHER__USE_WAREHOUSE": "false",
    }
    with patch.dict(os.environ, env, clear=True):
        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            warehouse_source = EarthRangerSourceConnection.from_named_connection("MEP_DEV").get_client()
            api_source = EarthRangerSourceConnection.from_named_connection("OTHER").get_client()

    assert isinstance(warehouse_source.warehouse, ERWarehouseClient)
    assert api_source.warehouse is None


def test_er_connection_use_warehouse_beats_the_global_flag(named_mock_env_with_token):
    env = named_mock_env_with_token | {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USE_WAREHOUSE": "false",
        "USE_EARTHRANGER_WAREHOUSE_API": "true",
    }
    with patch.dict(os.environ, env, clear=True):
        assert EarthRangerConnection.from_named_connection("MEP_DEV").warehouse_enabled() is False


@pytest.mark.parametrize("global_flag,expected", [("true", True), ("false", False)])
def test_er_connection_unset_use_warehouse_resolves_to_the_global_flag(
    named_mock_env_with_token, global_flag, expected
):
    """The connection resolves the tri-state, so the client it builds carries a verdict."""
    env = named_mock_env_with_token | {"USE_EARTHRANGER_WAREHOUSE_API": global_flag}
    with patch.dict(os.environ, env, clear=True):
        conn = EarthRangerSourceConnection.from_named_connection("MEP_DEV")
        assert conn.use_warehouse is None
        assert conn.warehouse_enabled() is expected

        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            source = conn.get_client()
    assert (source.warehouse is not None) is expected


def test_connection_field_validator_no_token():
    with pytest.raises(
        ValueError,
        match="If token is empty, EarthRanger username and password must be provided",
    ):
        EarthRangerConnection(
            server="https://test.com",
            username="username",
            tcp_limit="5",
            sub_page_size="4000",
        )


def test_connection_field_validator_token_and_creds():
    with pytest.raises(
        ValueError,
        match="Only a token, or an EarthRanger username and password can be provided, not both",
    ):
        EarthRangerConnection(
            server="https://test.com",
            username="username",
            password="password",
            token="123456",
            tcp_limit="5",
            sub_page_size="4000",
        )
