import os

import pandas as pd
import pytest

import ecoscope
from ecoscope.io.utils import TIME_COLS

pytestmark = pytest.mark.smart_io

SMART_SERVER = "https://maratriangleconnect.smartconservationtools.org/smartapi/"
SMART_USERNAME = os.getenv("SMART_USERNAME")
SMART_PASSWORD = os.getenv("SMART_PASSWORD")
CA_UUID = "735606d2-c34e-49c3-a45b-7496ca834e58"


def check_time_is_parsed(df):
    for col in TIME_COLS:
        if col in df.columns:
            assert pd.api.types.is_datetime64_ns_dtype(df[col]) or df[col].isna().all()


def test_smart_get_events(smart_io):
    events = smart_io.get_events(
        start="2021-12-20",
        end="2021-12-22",
    )
    assert not events.empty
    assert "time" in events.columns
    assert "event_type" in events.columns
    assert "geometry" in events.columns
    assert "extracted_attributes" in events.columns
    check_time_is_parsed(events)


def test_smart_get_patrol_observations(smart_io):
    result = smart_io.get_patrol_observations(
        start="2021-12-20",
        end="2021-12-22",
    )

    assert len(result.gdf) > 0
    assert "geometry" in result.gdf
    assert "groupby_col" in result.gdf
    assert "fixtime" in result.gdf
    check_time_is_parsed(result.gdf)


def test_smart_invalid_ca_raises():
    with pytest.raises(ValueError, match="Conservation area name 'Not a real CA' not found in API response"):
        ecoscope.io.SmartIO(
            urlBase=SMART_SERVER,
            username=SMART_USERNAME,
            password=SMART_PASSWORD,
            ca_name="Not a real CA",
        )


def test_smart_ca_id_resolution():
    smart_io = ecoscope.io.SmartIO(
        urlBase=SMART_SERVER,
        username=SMART_USERNAME,
        password=SMART_PASSWORD,
        ca_id="MTri",
    )
    assert smart_io._ca_uuid == CA_UUID


def test_smart_ca_name_resolution():
    smart_io = ecoscope.io.SmartIO(
        urlBase=SMART_SERVER,
        username=SMART_USERNAME,
        password=SMART_PASSWORD,
        ca_name="Mara Triangle Conservancy",
    )
    assert smart_io._ca_uuid == CA_UUID


def test_smart_language_code_resolution():
    smart_io = ecoscope.io.SmartIO(
        urlBase=SMART_SERVER,
        username=SMART_USERNAME,
        password=SMART_PASSWORD,
        ca_uuid=CA_UUID,
        language_code="en",
    )
    assert smart_io._language_uuid is not None
    assert smart_io._language_code == "en"


def test_smart_ca_resolution():
    # Provide all three identifiers - should succeed as the uuid is correct
    smart_io = ecoscope.io.SmartIO(
        urlBase=SMART_SERVER,
        username=SMART_USERNAME,
        password=SMART_PASSWORD,
        ca_uuid=CA_UUID,
        ca_id="wrong_id",
        ca_name="Wrong Name",
    )
    assert smart_io._ca_uuid == CA_UUID


def test_smart_invalid_language_code_raises():
    with pytest.raises(ValueError, match="Language code 'klingon' not found"):
        ecoscope.io.SmartIO(
            urlBase=SMART_SERVER,
            username=SMART_USERNAME,
            password=SMART_PASSWORD,
            ca_uuid=CA_UUID,
            language_code="klingon",
        )


def test_smart_no_ca_raises():
    with pytest.raises(
        ValueError, match="No conservation area identifier provided. Must provide either ca_uuid, ca_id, or ca_name"
    ):
        ecoscope.io.SmartIO(
            urlBase=SMART_SERVER,
            username=SMART_USERNAME,
            password=SMART_PASSWORD,
        )
