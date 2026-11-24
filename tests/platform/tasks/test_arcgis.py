"""fetch_arcgis_features and its read-only ArcGIS helper. No network: a fake HTTP session."""

import inspect
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from ecoscope.platform.tasks.io import ESRI_DISPLAY_FIELD, _arcgis, fetch_arcgis_features
from ecoscope.platform.tasks.io._arcgis import (
    ArcGISAuthError,
    ArcGISReadOnlyError,
    ReadOnlyArcGIS,
    features_to_frame,
)

LAYER = "https://services.example.com/ORG/arcgis/rest/services/Synthetic_Grid/FeatureServer/0"
TOKEN_URL = "https://www.arcgis.com/sharing/rest/generateToken"
MODULE = Path(_arcgis.__file__).resolve()

LAYER_INFO = {
    "id": 0,
    "serviceItemId": "abc123",
    "objectIdField": "OBJECTID",
    "globalIdField": "GlobalID",
    "maxRecordCount": 2,
    "editFieldsInfo": {"editDateField": "EditDate"},
    "fields": [
        {"name": "OBJECTID", "type": "esriFieldTypeOID"},
        {"name": "GlobalID", "type": "esriFieldTypeGlobalID"},
        {"name": "GridID", "type": "esriFieldTypeString"},
        {"name": "PlantingDate", "type": "esriFieldTypeDate"},
        {"name": "EditDate", "type": "esriFieldTypeDate"},
        {"name": "Shape__Area", "type": "esriFieldTypeDouble"},
    ],
}


def _feature(oid: int) -> dict[str, Any]:
    x = 10.0 + oid / 100
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[x, 1.0], [x + 0.001, 1.0], [x + 0.001, 1.001], [x, 1.0]]]},
        "properties": {
            "OBJECTID": oid,
            "GlobalID": "{AAAAAAAA-0000-4000-8000-%012d}" % oid,
            "GridID": f"G{oid:02d}",
            "PlantingDate": 1704067200000 if oid == 1 else None,  # 2024-01-01T00:00:00Z
            "EditDate": 1704153600000,  # 2024-01-02T00:00:00Z
            "Shape__Area": 100.0,
        },
    }


class FakeResponse:
    def __init__(self, body: dict[str, Any]):
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.body


class FakeSession:
    """Serves a 5-feature synthetic layer; records every request it is asked to send."""

    def __init__(self, n_features: int = 5, served: int | None = None, token_body: dict[str, Any] | None = None):
        self.features = [_feature(i) for i in range(1, n_features + 1)]
        self.served = self.features[: served if served is not None else n_features]
        self.token_body = token_body or {"token": "t0k3n"}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, timeout: int, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if url == TOKEN_URL:
            return FakeResponse(self.token_body)
        if url == LAYER:
            return FakeResponse(LAYER_INFO)
        params = kwargs["params"]
        if params.get("returnCountOnly") == "true":
            return FakeResponse({"count": len(self.features)})
        start, size = params["resultOffset"], params["resultRecordCount"]
        return FakeResponse({"type": "FeatureCollection", "features": self.served[start : start + size]})


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> FakeSession:
    fake = FakeSession()
    monkeypatch.setattr(_arcgis, "_new_session", lambda: fake)
    return fake


def _fetch(**kwargs: Any) -> Any:
    return fetch_arcgis_features(layer_url=LAYER + "/?f=json", username="viewer", password=SecretStr("pw"), **kwargs)


def test_fetch_pages_and_normalises(session: FakeSession) -> None:
    frame = _fetch()

    assert len(frame) == 5
    assert frame.crs.to_epsg() == 4326
    # maxRecordCount 2 → pages at offsets 0, 2, 4.
    offsets = [c[2]["params"]["resultOffset"] for c in session.calls if "resultOffset" in c[2].get("params", {})]
    assert offsets == [0, 2, 4]
    first = frame.iloc[0]
    assert first["esri_globalid"] == "aaaaaaaa-0000-4000-8000-000000000001"
    assert first["esri_source"] == "arcgis:abc123/0"
    assert first["esri_edit_date"] == "2024-01-02T00:00:00Z"
    assert first["PlantingDate"] == "2024-01-01T00:00:00Z"
    assert frame.iloc[1]["PlantingDate"] is None
    assert not frame["esri_filtered"].any()
    assert "Shape__Area" not in frame.columns
    assert list(frame["GridID"]) == ["G01", "G02", "G03", "G04", "G05"]


def test_fetch_only_sends_allowlisted_reads(session: FakeSession) -> None:
    _fetch()

    sent = {(method, url) for method, url, _ in session.calls}
    assert sent <= {("POST", TOKEN_URL), ("GET", LAYER), ("GET", LAYER + "/query")}
    assert ("POST", TOKEN_URL) in sent
    assert all("token" in c[2]["params"] for c in session.calls if c[0] == "GET")


@pytest.mark.parametrize(
    "method, suffix",
    [
        ("POST", "/applyEdits"),
        ("POST", "/addFeatures"),
        ("POST", "/updateFeatures"),
        ("POST", "/deleteFeatures"),
        ("POST", "/query"),
        ("DELETE", ""),
        ("PUT", ""),
        ("GET", "/../1/query"),
    ],
)
def test_helper_refuses_anything_but_reads(method: str, suffix: str) -> None:
    fake = FakeSession()
    arcgis = ReadOnlyArcGIS(LAYER, session=fake)

    with pytest.raises(ArcGISReadOnlyError):
        arcgis._request(method, LAYER + suffix)
    assert fake.calls == []


def test_helper_refuses_other_hosts() -> None:
    fake = FakeSession()
    arcgis = ReadOnlyArcGIS(LAYER, session=fake)

    with pytest.raises(ArcGISReadOnlyError):
        arcgis._request("GET", "https://evil.example.com/query")
    assert fake.calls == []


def test_module_contains_no_esri_edit_operations() -> None:
    edit_ops = re.compile(
        r"applyEdits|addFeatures|updateFeatures|deleteFeatures|calculate\b|truncate|/append\b|addToDefinition|updateDefinition",
        re.IGNORECASE,
    )
    offenders = [
        f"{path.name}:{n}"
        for path in [MODULE]
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if edit_ops.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == []


def test_partial_read_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_arcgis, "_new_session", lambda: FakeSession(served=3))

    with pytest.raises(RuntimeError, match="reported 5 features but returned 3"):
        _fetch()


def test_bad_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"error": {"code": 400, "message": "Invalid username or password."}}
    monkeypatch.setattr(_arcgis, "_new_session", lambda: FakeSession(token_body=body))

    with pytest.raises(ArcGISAuthError, match="Invalid username or password"):
        _fetch()


def test_filtered_read_is_flagged(session: FakeSession) -> None:
    frame = _fetch(where="GridID LIKE 'G%'")

    assert frame["esri_filtered"].all()


def test_layer_without_globalid_raises(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    monkeypatch.setitem(LAYER_INFO, "globalIdField", None)

    with pytest.raises(ValueError, match="no GlobalID field"):
        _fetch()


def test_empty_layer_returns_typed_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_arcgis, "_new_session", lambda: FakeSession(n_features=0))

    frame = _fetch()

    assert len(frame) == 0
    assert {"GridID", "esri_globalid", "esri_source", "esri_filtered", "geometry"} <= set(frame.columns)


def test_rejects_non_layer_url() -> None:
    with pytest.raises(ValueError, match="FeatureServer"):
        ReadOnlyArcGIS("https://services.example.com/ORG/arcgis/rest/services/Synthetic_Grid/FeatureServer")


def _features() -> list[dict[str, Any]]:
    return [_feature(i) for i in range(1, 6)]


def test_frame_carries_display_field() -> None:
    with_field = features_to_frame(_features(), {**LAYER_INFO, "displayField": "PageName"}, "arcgis:abc123/0", False)
    without = features_to_frame(_features(), LAYER_INFO, "arcgis:abc123/0", False)

    assert with_field[ESRI_DISPLAY_FIELD].unique().tolist() == ["PageName"]
    assert without[ESRI_DISPLAY_FIELD].unique().tolist() == [None]


def test_display_field_reserved_name_clash() -> None:
    info = {
        **LAYER_INFO,
        "fields": [*LAYER_INFO["fields"], {"name": "esri_display_field", "type": "esriFieldTypeString"}],
    }

    with pytest.raises(ValueError, match="reserves"):
        features_to_frame(_features(), info, "arcgis:abc123/0", False)


def test_password_help_text() -> None:
    field = fetch_arcgis_features.model_fields["password"] if hasattr(fetch_arcgis_features, "model_fields") else None
    description = (
        field.description
        if field is not None
        else inspect.signature(fetch_arcgis_features).parameters["password"].annotation.__metadata__[0].description
    )

    assert description == "Stored with this setup on the platform. A Viewer account is enough."


def test_messages_say_arcgis() -> None:
    # Everything after the module docstring (developer-facing) except comments: the task docstring,
    # Field texts and error messages are all user-facing. `esri_*` identifiers don't match \bEsri\b.
    source = MODULE.read_text().split('"""', 2)[2]
    lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]

    assert [line for line in lines if re.search(r"\bEsri\b", line)] == []
