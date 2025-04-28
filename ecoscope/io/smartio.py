import json
import logging
import uuid
from datetime import timedelta

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import requests
import shapely

import ecoscope

logger = logging.getLogger(__name__)


class SmartIO:
    def __init__(self, **kwargs):
        self._urlBase = kwargs.get("urlBase")
        self._username = kwargs.get("username")
        self._password = kwargs.get("password")

        self._session = requests.Session()
        self._token = kwargs.get("token")
        self._verify_ssl = kwargs.get("verify_ssl")

        self.login()

    def login(self) -> None:
        login_data = {
            "username": self._username,
            "password": self._password,
        }

        if not self._token:
            self._session = requests.Session()
            response = self._session.post(f"{self._urlBase}token", data=login_data, verify=self._verify_ssl)

            response.raise_for_status()
            self._token = response.json()["access_token"]
            return

    def query_data(self, url: str, params: dict | None = None) -> pd.DataFrame:
        headers = {
            "Authorization": f"Bearer {self._token}",
        }

        if params is None:
            params = {}

        session = requests.Session()
        r = session.get(
            f"{self._urlBase}{url}",
            verify=self._verify_ssl,
            params=params,
            headers=headers,
        )

        r.raise_for_status()
        return pd.DataFrame(r.json())

    def query_geojson_data(self, url: str, params: dict | None = None) -> gpd.GeoDataFrame | None:
        headers = {
            "Authorization": f"Bearer {self._token}",
        }

        if params is None:
            params = {}

        session = requests.Session()
        r = session.get(
            f"{self._urlBase}{url}",
            verify=self._verify_ssl,
            params=params,
            headers=headers,
        )

        r.raise_for_status()
        return gpd.GeoDataFrame.from_features(r.json(), crs=4326)

    def get_patrols_list(
        self,
        ca_uuid: str,
        language_uuid: str,
        start: str,
        end: str,
        patrol_mandate: str | None,
        patrol_transport: str | None,
    ) -> gpd.GeoDataFrame | None:
        params: dict[str, str | None] = {}
        params["ca_uuid"] = ca_uuid
        params["language_uuid"] = language_uuid
        params["start_date"] = start
        params["end_date"] = end
        params["patrol_mandate"] = patrol_mandate
        params["patrol_transport"] = patrol_transport

        return self.query_geojson_data(url="patrol/", params=params)

    def extract_coordinates(self, gdf):
        """
        Extract coordinates and timestamps from a GeoDataFrame with MultiLineString geometries.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame with MultiLineString geometries

        Returns:
            tuple: Three lists containing longitudes, latitudes, and timestamps
        """
        longitudes = []
        latitudes = []
        timestamps = []

        if isinstance(gdf, gpd.GeoDataFrame):
            geometries = gdf["geometry"]
        elif isinstance(gdf, (shapely.geometry.MultiLineString, shapely.geometry.LineString)):
            geometries = [gdf]
        else:
            geometries = gdf

        for geometry in geometries:
            try:
                # Handle MultiLineString
                if isinstance(geometry, shapely.geometry.MultiLineString):
                    geoms_to_iterate = geometry.geoms
                # Handle single LineString
                elif isinstance(geometry, shapely.geometry.LineString):
                    geoms_to_iterate = [geometry]
                else:
                    continue

                for linestring in geoms_to_iterate:
                    for coord in linestring.coords:
                        if len(coord) >= 3:  # Ensure we have x, y, and timestamp
                            longitudes.append(coord[0])
                            latitudes.append(coord[1])
                            timestamps.append(coord[2])
            except Exception as e:
                logger.error(f"Error processing geometry: {e}")

        return longitudes, latitudes, timestamps

    def process_patrols_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Process multiple geometries in a vectorized way.
        Args:
            df: Input DataFrame with geometry column containing MULTILINESTRING Z data
        Returns:
        gpd.GeoDataFrame: Processed GeoDataFrame with expanded coordinate data
        """
        if df.empty:
            return gpd.GeoDataFrame()

        all_coords = []

        for index, row in df.iterrows():
            geometry = row["geometry"]
            if geometry is None:
                continue

            longitudes, latitudes, timestamps = self.extract_coordinates(geometry)

            if not longitudes:
                logger.error(f"Warning: No valid coordinates found in geometry at index {index}")
                continue

            times = pd.to_datetime(timestamps, unit="ms", utc=True)

            coords_data = pd.DataFrame({"longitude": longitudes, "latitude": latitudes, "fixtime": times})

            static_data = {col: row[col] for col in df.columns}
            for col, value in static_data.items():
                coords_data[col] = value

            all_coords.append(coords_data)

        if not all_coords:
            return gpd.GeoDataFrame()

        result = pd.concat(all_coords, ignore_index=True)
        result_df = gpd.GeoDataFrame(
            result,
            geometry=gpd.points_from_xy(x=result["longitude"], y=result["latitude"]),
            crs="EPSG:4326",
        )
        result_df = result_df.rename(
            columns={
                "uuid": "patrol_id",
                "patrol_leg_day_start": "patrol_start_time",
                "patrol_leg_day_end": "patrol_end_time",
                "id": "groupby_col",
            }
        )
        result_df["patrol_type__display"] = result_df["patrol_mandate"]

        return result_df

    def get_patrol_observations(
        self,
        ca_uuid: str,
        language_uuid: str,
        start: str,
        end: str,
        patrol_mandate: str | None = None,
        patrol_transport: str | None = None,
        window_size_in_days: int = 7,
    ) -> ecoscope.Relocations | None:
        df = gpd.GeoDataFrame()
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        total_duration = end_dt - start_dt

        if total_duration <= timedelta(days=window_size_in_days):
            df = self.get_patrols_list(
                ca_uuid=ca_uuid,
                language_uuid=language_uuid,
                start=start,
                end=end,
                patrol_mandate=patrol_mandate,
                patrol_transport=patrol_transport,
            )
        else:
            current_start = start_dt
            while current_start < end_dt:
                segment_end = min(current_start + timedelta(days=window_size_in_days), end_dt)
                patrols = self.get_patrols_list(
                    ca_uuid=ca_uuid,
                    language_uuid=language_uuid,
                    start=current_start.isoformat(),
                    end=segment_end.isoformat(),
                    patrol_mandate=patrol_mandate,
                    patrol_transport=patrol_transport,
                )
                df = pd.concat([df, patrols])
                current_start = segment_end

        try:
            patrols_df = self.process_patrols_gdf(df)

            if patrols_df.empty:
                return None

            patrols_relocs = ecoscope.Relocations.from_gdf(
                patrols_df,
                groupby_col="groupby_col",
                uuid_col="patrol_id",
                time_col="fixtime",
            )
            patrols_relocs.gdf.reset_index()
            patrols_relocs.gdf.columns = [col.replace("extra__", "") for col in patrols_relocs.gdf.columns]
            patrols_relocs.gdf = patrols_relocs.gdf.assign(
                id=[str(uuid.uuid4()) for _ in range(len(patrols_relocs.gdf))]
            ).set_index("id")
            return patrols_relocs
        except Exception as e:
            logger.error(f"Error processing patrol data: {e}")
            return None

    def get_events(
        self,
        ca_uuid: str,
        language_uuid: str,
        start: str,
        end: str,
    ) -> gpd.GeoDataFrame:
        params = {}
        params["ca_uuid"] = ca_uuid
        params["language_uuid"] = language_uuid
        params["start_date"] = start
        params["end_date"] = end

        events_df = self.query_geojson_data(url="observation/", params=params)
        events_df = gpd.GeoDataFrame(
            events_df,
            geometry=gpd.points_from_xy(x=events_df["X"], y=events_df["Y"]),  # type: ignore[index]
            crs="EPSG:4326",
        )

        if "datetime" in events_df.columns:
            events_df = events_df.rename(columns={"datetime": "time"})
        else:
            logger.warning('"datetime" column does not exist in events_df')

        if "category" in events_df.columns:
            events_df = events_df.rename(columns={"category": "event_type"})
        else:
            logger.warning('"category" column does not exist in events_df')

        events_df["extracted_attributes"] = events_df["attributes"].apply(self.extract_event_attributes)
        return events_df

    def extract_event_attributes(self, attr_str: str) -> dict:
        attr_list = json.loads(attr_str)
        values = {}
        for attr in attr_list:
            values[attr["attribute"]] = attr["value"]

        return values
