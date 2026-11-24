"""Read an ArcGIS Feature Service layer — strictly read-only.

Every request to ArcGIS goes through `ReadOnlyArcGIS`, which allows exactly three calls and
refuses anything else before a request is sent:

- ``POST <portal>/sharing/rest/generateToken`` (ArcGIS requires POST for credentials)
- ``GET <layer>`` (layer metadata)
- ``GET <layer>/query`` (feature reads and the record count)

Nothing in this module builds an ArcGIS edit call. ``tests/platform/tasks/test_arcgis.py`` proves the
allowlist and scans this module's source for Esri edit operation names.
"""

import re
from datetime import datetime, timezone
from typing import Annotated, Any, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame

ESRI_SOURCE = "esri_source"
ESRI_GLOBALID = "esri_globalid"
ESRI_EDIT_DATE = "esri_edit_date"
ESRI_FILTERED = "esri_filtered"
ESRI_DISPLAY_FIELD = "esri_display_field"

DEFAULT_PORTAL_URL = "https://www.arcgis.com"
ALL_FEATURES = "1=1"
_LAYER_URL = re.compile(r"/(FeatureServer|MapServer)/\d+$", re.IGNORECASE)
_GUID_BRACES = re.compile(r"[{}]")
# Esri-maintained geometry measures; they describe the Esri shape, not the feature.
_DROPPED_FIELD_PREFIX = "shape__"
_AUTH_ERROR_CODES = {498, 499}
_TIMEOUT_S = 120


class ArcGISReadOnlyError(RuntimeError):
    """Raised before sending any ArcGIS request outside the read-only allowlist."""


class ArcGISAuthError(RuntimeError):
    """Raised when ArcGIS rejects the credentials or the token."""


def normalise_layer_url(layer_url: str) -> str:
    """Strip query, fragment and trailing slash; require a ``…/FeatureServer/<n>`` layer URL."""
    parts = urlsplit(layer_url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Not a URL: {layer_url!r}")
    path = parts.path.rstrip("/")
    if not _LAYER_URL.search(path):
        raise ValueError(f"Expected a feature layer URL ending in /FeatureServer/<layer number>, got {layer_url!r}")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _new_session() -> Any:
    import requests

    return requests.Session()


class ReadOnlyArcGIS:
    """The only way this module talks to ArcGIS. See the module docstring for the allowlist."""

    def __init__(self, layer_url: str, portal_url: str = DEFAULT_PORTAL_URL, session: Any = None):
        self.layer_url = normalise_layer_url(layer_url)
        self.token_url = portal_url.strip().rstrip("/") + "/sharing/rest/generateToken"
        self._allowed = {
            ("POST", self.token_url),
            ("GET", self.layer_url),
            ("GET", self.layer_url + "/query"),
        }
        self._session = session if session is not None else _new_session()
        self._token: str | None = None

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if (method.upper(), url) not in self._allowed:
            raise ArcGISReadOnlyError(f"Refusing {method} {url}: this task only reads from ArcGIS.")
        import requests

        try:
            response = self._session.request(method.upper(), url, timeout=_TIMEOUT_S, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            # requests' own messages embed the full URL, query string (and so the token) included.
            status = exc.response.status_code if exc.response is not None else type(exc).__name__
            raise RuntimeError(f"ArcGIS {method.upper()} {url} failed ({status}).") from None
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if error:
            code, message = error.get("code"), error.get("message", "unknown error")
            if code in _AUTH_ERROR_CODES or url == self.token_url:
                raise ArcGISAuthError(f"ArcGIS rejected the request ({code}): {message}")
            raise RuntimeError(f"ArcGIS returned an error ({code}): {message} {error.get('details', '')}")
        return cast(dict[str, Any], body)

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", url, params={**params, "token": self._token})

    def authenticate(self, username: str, password: str) -> None:
        body = self._request(
            "POST",
            self.token_url,
            data={
                "username": username,
                "password": password,
                "referer": DEFAULT_PORTAL_URL,
                "expiration": 60,
                "f": "json",
            },
        )
        if "token" not in body:
            raise ArcGISAuthError("ArcGIS did not return a token; check the username, password and portal URL.")
        self._token = body["token"]

    def layer_info(self) -> dict[str, Any]:
        return self._get(self.layer_url, {"f": "json"})

    def count(self, where: str) -> int:
        return int(
            self._get(self.layer_url + "/query", {"where": where, "returnCountOnly": "true", "f": "json"})["count"]
        )

    def query_page(self, where: str, order_by: str, offset: int, page_size: int) -> dict[str, Any]:
        return self._get(
            self.layer_url + "/query",
            {
                "where": where,
                "outFields": "*",
                "outSR": 4326,
                "returnGeometry": "true",
                "orderByFields": order_by,
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "f": "geojson",
            },
        )


def esri_source_id(layer_url: str, layer_info: dict[str, Any]) -> str:
    """A stable id for the layer: its ArcGIS item id and layer number, else the normalised URL."""
    item_id, layer_id = layer_info.get("serviceItemId"), layer_info.get("id")
    if item_id and layer_id is not None:
        return f"arcgis:{item_id}/{layer_id}"
    return "arcgis:" + normalise_layer_url(layer_url).lower()


def _normalise_guid(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = _GUID_BRACES.sub("", str(value)).strip().lower()
    return text or None


def _epoch_ms_to_iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _empty_frame(attribute_columns: list[str]) -> AnyGeoDataFrame:
    import geopandas as gpd  # type: ignore[import-untyped]

    columns = attribute_columns + [ESRI_SOURCE, ESRI_GLOBALID, ESRI_EDIT_DATE, ESRI_FILTERED, ESRI_DISPLAY_FIELD]
    return cast(AnyGeoDataFrame, gpd.GeoDataFrame(columns=columns, geometry=[], crs="EPSG:4326"))


def features_to_frame(
    features: list[dict[str, Any]], layer_info: dict[str, Any], source_id: str, filtered: bool
) -> AnyGeoDataFrame:
    """GeoJSON features (EPSG:4326) → one row per feature with normalised ``esri_*`` columns."""
    import geopandas as gpd  # type: ignore[import-untyped]

    fields = layer_info.get("fields", [])
    globalid_field = layer_info.get("globalIdField")
    if not globalid_field:
        raise ValueError(
            "This layer has no GlobalID field, so features cannot be matched safely between runs. "
            "Enable GlobalIDs on the layer in ArcGIS, or ask for a different match key."
        )
    edit_date_field = (layer_info.get("editFieldsInfo") or {}).get("editDateField")
    date_fields = {f["name"] for f in fields if f.get("type") == "esriFieldTypeDate"}
    attribute_columns = [f["name"] for f in fields if not f["name"].lower().startswith(_DROPPED_FIELD_PREFIX)]
    clashes = {ESRI_SOURCE, ESRI_GLOBALID, ESRI_EDIT_DATE, ESRI_FILTERED, ESRI_DISPLAY_FIELD} & set(attribute_columns)
    if clashes:
        raise ValueError(f"Layer fields clash with names this task reserves: {sorted(clashes)}")
    if not features:
        return _empty_frame(attribute_columns)

    rows, geometries = [], []
    for feature in features:
        props = feature.get("properties") or {}
        row = {}
        for name in attribute_columns:
            value = props.get(name)
            row[name] = _epoch_ms_to_iso(value) if name in date_fields else value
        row[ESRI_SOURCE] = source_id
        row[ESRI_GLOBALID] = _normalise_guid(props.get(globalid_field))
        row[ESRI_EDIT_DATE] = _epoch_ms_to_iso(props.get(edit_date_field)) if edit_date_field else None
        row[ESRI_FILTERED] = filtered
        row[ESRI_DISPLAY_FIELD] = layer_info.get("displayField") or None
        rows.append(row)
        geometries.append(feature.get("geometry"))

    frame = gpd.GeoDataFrame(rows, geometry=_shapes(geometries), crs="EPSG:4326")
    missing = frame[ESRI_GLOBALID].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} ArcGIS feature(s) have no GlobalID; refusing to sync.")
    duplicated = frame[ESRI_GLOBALID].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"Duplicate GlobalIDs in the ArcGIS layer: {sorted(set(frame.loc[duplicated, ESRI_GLOBALID]))[:5]}"
        )
    return cast(AnyGeoDataFrame, frame)


def _shapes(geometries: list[dict[str, Any] | None]) -> list[Any]:
    from shapely.geometry import shape

    return [shape(g) if g else None for g in geometries]


@register(tags=["io"])
def fetch_arcgis_features(
    layer_url: Annotated[
        str,
        Field(
            title="Feature layer URL",
            description="The layer's REST URL, ending in /FeatureServer/<layer number>.",
        ),
    ],
    username: Annotated[str, Field(title="ArcGIS username")],
    password: Annotated[
        SecretStr,
        Field(
            title="ArcGIS password",
            description="Stored with this setup on the platform. A Viewer account is enough.",
        ),
    ],
    portal_url: Annotated[
        str,
        AdvancedField(
            default=DEFAULT_PORTAL_URL,
            title="Portal URL",
            description="ArcGIS Online, or your ArcGIS Enterprise portal.",
        ),
    ] = DEFAULT_PORTAL_URL,
    where: Annotated[
        str,
        AdvancedField(
            default=ALL_FEATURES,
            title="Filter (SQL where)",
            description="Leave as 1=1 to sync every feature. Any other filter switches deletes off.",
        ),
    ] = ALL_FEATURES,
) -> AnyGeoDataFrame:
    """Download every feature of an ArcGIS layer, read-only, as EPSG:4326 polygons/lines/points.

    Returns one row per feature: the layer's attributes (dates as ISO 8601 UTC strings, ArcGIS
    ``Shape__*`` measures dropped) plus ``esri_source`` (``arcgis:<item id>/<layer>``),
    ``esri_globalid`` (lower-case, no braces), ``esri_edit_date``, ``esri_filtered`` and
    ``esri_display_field`` (the layer's displayField, or None).
    Raises rather than returning a partial layer: the record count is checked against the
    features received, because a short read would look like deleted features.
    """
    where = where.strip() or ALL_FEATURES
    arcgis = ReadOnlyArcGIS(layer_url, portal_url)
    arcgis.authenticate(username, password.get_secret_value())
    info = arcgis.layer_info()
    source_id = esri_source_id(layer_url, info)
    expected = arcgis.count(where)
    page_size = int(info.get("maxRecordCount") or 1000)
    order_by = info.get("objectIdField") or "OBJECTID"

    features: list[dict[str, Any]] = []
    while len(features) < expected:
        page = arcgis.query_page(where, order_by, offset=len(features), page_size=page_size)
        batch = page.get("features") or []
        if not batch:
            break
        features.extend(batch)

    if len(features) != expected:
        raise RuntimeError(
            f"ArcGIS reported {expected} features but returned {len(features)}; refusing to sync a partial layer."
        )
    return features_to_frame(features, info, source_id, filtered=where != ALL_FEATURES)
