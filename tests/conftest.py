import math
import os
import warnings
from dataclasses import dataclass

import ee
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

import ecoscope
from ecoscope import Relocations

os.environ["USE_PYGEOS"] = "0"


_METERS_PER_DEGREE = 111_320.0  # approx metres per degree of latitude


@dataclass(frozen=True)
class Segment:
    """A synthetic trajectory segment for tests: a start point, a time window, a distance,
    and a bearing.

    ``end_point`` is derived so the drawn line genuinely spans ``dist_meters`` along
    ``bearing_deg`` (0 = north, 90 = east), keeping the geometry self-consistent with the
    stated distance rather than using an arbitrary fixed step. Assemble a list of these into
    a GeoDataFrame with ``build_segments``.
    """

    start_point: Point
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    dist_meters: float = 1000.0
    bearing_deg: float = 45.0

    @property
    def end_point(self) -> Point:
        span_deg = self.dist_meters / _METERS_PER_DEGREE
        theta = math.radians(self.bearing_deg)
        dlat = span_deg * math.cos(theta)
        dlon = span_deg * math.sin(theta) / math.cos(math.radians(self.start_point.y))
        return Point(self.start_point.x + dlon, self.start_point.y + dlat)

    def to_row(self) -> dict:
        return {
            "segment_start": pd.Timestamp(self.start_time),
            "segment_end": pd.Timestamp(self.end_time),
            "geometry": LineString([self.start_point, self.end_point]),
            "dist_meters": float(self.dist_meters),
        }


def build_segments(segments, crs="EPSG:4326"):
    """Assemble Segment instances into a trajectory-segment GeoDataFrame."""
    return gpd.GeoDataFrame([s.to_row() for s in segments], crs=crs)


# Named start locations for building synthetic trajectory segments.
EQUATOR = Point(0.0, 0.0)
ARCTIC = Point(0.0, 80.0)


def pytest_configure(config):
    ecoscope.init()

    os.makedirs("tests/outputs", exist_ok=True)

    if "io" in config.inicfg.get("markers"):
        try:
            EE_ACCOUNT = os.getenv("EE_ACCOUNT")
            EE_PRIVATE_KEY_DATA = os.getenv("EE_PRIVATE_KEY_DATA")
            if EE_ACCOUNT and EE_PRIVATE_KEY_DATA:
                credentials = ee.ServiceAccountCredentials(EE_ACCOUNT, key_data=EE_PRIVATE_KEY_DATA)
                ee.Initialize(credentials=credentials)
                pytest.earthengine = True
        except Exception:
            warnings.warn(Warning("Earth Engine can not be initialized."))


@pytest.fixture(scope="session")
def er_io():
    ER_SERVER = "https://mep-dev.pamdas.org"
    ER_USERNAME = os.getenv("ER_USERNAME")
    ER_PASSWORD = os.getenv("ER_PASSWORD")
    er_io = ecoscope.io.EarthRangerIO(server=ER_SERVER, username=ER_USERNAME, password=ER_PASSWORD)

    er_io.GROUP_NAME = "Elephants"
    er_io.SUBJECT_IDS = ["64444ed7-72ec-4531-a2b1-fb25c7197b2d", "b8be28f7-8c20-46d9-85a5-fd817351bde5"]
    er_io.SUBJECTSOURCE_IDS = ["893d4255-51e1-4567-8ac1-028e7c532431", "59568b94-b1c8-4f19-89a7-3d08ae60e52b"]
    er_io.SOURCE_IDS = ["84f240db-dd7b-4297-bf85-8c051b8055c7", "7e5b3e3b-3d1c-481c-905e-2eb9454c6a31"]
    return er_io


@pytest.fixture(scope="session")
def smart_io():
    SMART_SERVER = "https://maratriangleconnect.smartconservationtools.org/smartapi/"
    SMART_USERNAME = os.getenv("SMART_USERNAME")
    SMART_PASSWORD = os.getenv("SMART_PASSWORD")
    smart_io = ecoscope.io.SmartIO(urlBase=SMART_SERVER, username=SMART_USERNAME, password=SMART_PASSWORD)

    return smart_io


@pytest.fixture(scope="session")
def er_events_io():
    ER_SERVER = "https://mep-dev.pamdas.org"
    ER_USERNAME = os.getenv("ER_USERNAME")
    ER_PASSWORD = os.getenv("ER_PASSWORD")
    er_events_io = ecoscope.io.EarthRangerIO(
        server=ER_SERVER, username=ER_USERNAME, password=ER_PASSWORD, tcp_limit=5, sub_page_size=100
    )

    return er_events_io


@pytest.fixture(scope="session")
def movebank_gdf():
    df = pd.read_feather("tests/sample_data/vector/movebank_data.feather")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.pop("location-long"), df.pop("location-lat")),
        crs=4326,
    )
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
    return gdf


@pytest.fixture
def movebank_relocations(movebank_gdf):
    return Relocations.from_gdf(
        movebank_gdf,
        groupby_col="individual-local-identifier",
        time_col="timestamp",
        uuid_col="event-id",
    )


@pytest.fixture(scope="session")
def aoi_gdf():
    AOI_FILE = "tests/sample_data/vector/maec_4zones_UTM36S.gpkg"
    regions_gdf = gpd.GeoDataFrame.from_file(AOI_FILE).to_crs(4326)
    regions_gdf.set_index("ZONE", drop=True, inplace=True)
    return regions_gdf


@pytest.fixture(scope="session")
def sample_relocs_gdf():
    gdf = gpd.read_parquet("tests/sample_data/vector/sample_relocs.parquet")
    return ecoscope.io.utils.clean_time_cols(gdf)


@pytest.fixture
def sample_relocs(sample_relocs_gdf):
    # Relocations.from_gdf defaults to copy=True, so the cached session gdf is not mutated.
    return ecoscope.Relocations.from_gdf(sample_relocs_gdf)


@pytest.fixture
def sample_events_df_with_bad_geojson():
    """
    A mock get_events response with intentionally bad geojson:
    There are 6 events in this mock
    event 0: 'geometry' is None
    event 5: 'geomtery' and 'properties' are None
    """
    return pd.read_feather("tests/sample_data/io/get_events_bad_geojson.feather")


@pytest.fixture
def sample_patrol_events_with_bad_geojson():
    """
    A mock get_patrol_events response with intentionally bad geojson:
    There's a single patrol in this mock with events that have the following problems in their json
        event 0: 'geometry' key is not present
        event 1: 'properties' key is not present
        event 2: 'datetime' key is not present within 'properties
        event 3: is untouched
        event 4: 'geojson' is an empty dict
    """
    return pd.read_json("tests/sample_data/io/get_patrol_events_bad_geojson.json")


@pytest.fixture
def sample_patrol_events_with_poly():
    """
    A mock get_patrol_events response that includes non-point geometry
    """
    return pd.read_json("tests/sample_data/io/get_patrol_events_with_poly_geojson.json")
