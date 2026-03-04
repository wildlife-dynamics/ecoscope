import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, validate_call

from ecoscope.platform.annotations import DataFrame
from ecoscope.platform.connections import (
    EarthEngineClient,
    EarthEngineConnection,
    EarthRangerClient,
    EarthRangerConnection,
    SmartClient,
    SmartConnection,
    is_client,
)


def test_is_client():
    assert is_client(EarthRangerClient)
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
        with patch("ecoscope.io.EarthRangerIO", autospec=True):
            client = f(client="mep_dev")
            assert hasattr(client, "get_subjectgroup_observations")
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
