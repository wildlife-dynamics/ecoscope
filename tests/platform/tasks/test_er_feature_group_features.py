import json
from typing import Any

import pytest
from shapely.geometry import MultiPolygon, Polygon, mapping

from ecoscope.platform.connections import EarthRangerBackends
from ecoscope.platform.tasks.io import get_er_feature_group_features


def _member(
    er_id: str, name: str, attrs: dict[str, Any], i: int, geom: Any = None
) -> dict[str, Any]:
    geom = geom or MultiPolygon([Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)])])
    return {
        "type": "Feature",
        "id": er_id,
        "properties": {
            "name": name,
            "feature_type": "type-1",
            "attributes": attrs,
            "pk": er_id,
        },
        "geometry": json.loads(json.dumps(mapping(geom))),
    }


class FakeAPI:
    """The A1 read methods, recording every call."""

    def __init__(self, groups: dict[str, dict[str, Any]]):
        self.groups = groups  # id -> {"name", "members": [GeoJSON features]}
        self.calls: list[tuple[str, ...]] = []

    def get_feature_groups(self) -> list[dict[str, Any]]:
        self.calls.append(("get_feature_groups",))
        return [
            {"id": k, "name": g["name"], "feature_count": len(g["members"])}
            for k, g in self.groups.items()
        ]

    def get_feature_group(self, group_id: str) -> dict[str, Any]:
        self.calls.append(("get_feature_group", group_id))
        members = self.groups[group_id]["members"]
        return {
            "id": group_id,
            "features": [{"type": "FeatureCollection", "features": members}]
            if members
            else [],
        }


def _read(api: FakeAPI, name: str):
    return get_er_feature_group_features(
        EarthRangerBackends(api=api, warehouse=None), name
    )  # type: ignore[arg-type]


def _synced(letter: str, i: int, er_id: str) -> dict[str, Any]:
    attrs = {
        "GridID": f"Grid-{letter}",
        "esri_source": "arcgis:synthetic/0",
        "esri_globalid": f"{{{letter}}}",
        "esri_edit_date": "2026-01-01T00:00:00Z",
    }
    return _member(er_id, f"Grid-{letter}", attrs, i)


def test_missing_group_returns_empty_with_columns() -> None:
    out = _read(FakeAPI({}), "Nope")

    assert len(out) == 0
    assert list(out.columns) == [
        "er_id", "name", "feature_type", "esri_source", "esri_globalid", "esri_edit_date", "attributes_json", "geometry",
    ]  # fmt: skip


def test_reads_members_and_sync_stamps() -> None:
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])
    api = FakeAPI(
        {
            "g1": {
                "name": "Grid",
                "members": [
                    _synced("A", 0, "er-a"),
                    _member("er-h", "Hand", {}, 7, poly),
                ],
            }
        }
    )

    out = _read(api, " Grid ")

    assert list(out.er_id) == ["er-a", "er-h"]
    assert out.loc[0, "esri_globalid"] == "{A}"
    assert out.loc[0, "esri_source"] == "arcgis:synthetic/0"
    assert out.loc[1, "esri_globalid"] is None
    attrs = json.loads(out.loc[0, "attributes_json"])
    assert out.loc[0, "attributes_json"] == json.dumps(attrs, sort_keys=True)
    assert out.loc[0, "name"] == "Grid-A"
    assert out.geometry.iloc[0].geom_type == "MultiPolygon"
    assert out.geometry.iloc[1].geom_type == "Polygon"
    assert out.crs.to_epsg() == 4326


def test_reads_only_target_group() -> None:
    api = FakeAPI(
        {
            "g1": {"name": "Grid", "members": [_synced("A", 0, "er-a")]},
            "g2": {"name": "Other", "members": [_synced("A", 0, "er-a")]},
        }
    )

    _read(api, "Grid")

    assert api.calls == [("get_feature_groups",), ("get_feature_group", "g1")]


def test_duplicate_group_names_raise() -> None:
    api = FakeAPI(
        {"g1": {"name": "Dup", "members": []}, "g2": {"name": "Dup", "members": []}}
    )

    with pytest.raises(
        ValueError, match="2 EarthRanger feature groups are named 'Dup'"
    ):
        _read(api, "Dup")
