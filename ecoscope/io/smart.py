import requests
import pandas as pd
import geopandas as gpd


class smart_api:
    def __init__(
        self, urlBase: str, username: str, password: str, ca_uuid: str, language_uuid: str, verify_ssl: bool = False
    ):
        self._urlBase = urlBase
        self._username = username
        self._password = password
        self._ca_uuid = ca_uuid
        self._language_uuid = language_uuid
        self._session = requests.Session()
        self._token = None
        self._verify_ssl = verify_ssl

    def login(self):
        login_data = {
            "username": self._username,
            "password": self._password,
        }
        # print(self._urlBase)

        if not self._token:
            self._session = requests.Session()
            response = self._session.post(f"{self._urlBase}token", data=login_data, verify=self._verify_ssl)
            # print(response)
            if response.status_code == 200:
                self._token = response.json()["access_token"]
                return
            else:
                self._token = None
                self._session = requests.Session()
                raise Exception("failed to login")

    def query_data(self, url, params={}):
        headers = {
            "Authorization": f"Bearer {self._token}",
        }

        session = requests.Session()
        r = session.get(f"{self._urlBase}{url}", verify=self._verify_ssl, params=params, headers=headers)

        if r.status_code == 200:
            df = pd.DataFrame(r.json())
        else:
            print(r.status_code)
            df = None

        return df

    def query_geojson_data(self, url, params={}) -> gpd.GeoDataFrame | None:
        headers = {
            "Authorization": f"Bearer {self._token}",
        }

        session = requests.Session()
        r = session.get(f"{self._urlBase}{url}", verify=self._verify_ssl, params=params, headers=headers)

        if r.status_code == 200:
            df = gpd.GeoDataFrame.from_features(r.json(), crs=4326)
        else:
            print(r.status_code)
            df = None
        return df

    def get(self, url, params={}):
        headers = {
            "Authorization": f"Bearer {self._token}",
        }

        session = requests.Session()
        r = session.get(f"{self._urlBase}{url}", verify=self._verify_ssl, params=params, headers=headers)
        return r
