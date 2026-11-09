import pyarrow as pa
import pytest

from ecoscope.platform.data_sets import (
    TABLE_TYPES,
    DataSetManifest,
    DataSetSchemaError,
    table_filename,
    validate_table,
)


def test_registry_has_the_five_initial_table_types():
    assert set(TABLE_TYPES) == {
        "patrol_tracks",
        "patrol_events",
        "patrol_legs",
        "spatial_feature_groups",
        "events",
    }
    assert all(t.schema_version == "1" for t in TABLE_TYPES.values())
    assert TABLE_TYPES["patrol_legs"].spatial is False
    assert TABLE_TYPES["patrol_tracks"].spatial is True


def test_validate_table_accepts_extra_columns_in_any_order(
    spatial_feature_groups_table,
):
    extra = spatial_feature_groups_table.append_column("note", pa.array(["x"] * spatial_feature_groups_table.num_rows))
    validate_table(extra.select(sorted(extra.column_names)), "spatial_feature_groups")


def test_validate_table_names_missing_fields():
    t = pa.table({"name": pa.array(["a"])})
    with pytest.raises(DataSetSchemaError, match=r"spatial_feature_groups.*missing.*pk"):
        validate_table(t, "spatial_feature_groups")


def test_validate_table_rejects_unknown_schema_version(spatial_feature_groups_table):
    with pytest.raises(DataSetSchemaError, match=r"schema_version.*expected 1.*found 9"):
        validate_table(spatial_feature_groups_table, "spatial_feature_groups", schema_version="9")


def test_manifest_round_trips_through_json():
    m = DataSetManifest.model_validate(
        {
            "data_set": "patrols",
            "tables": {
                "patrol_legs": {
                    "path": "data_sets/patrol_legs.arrow",
                    "rows": 3,
                    "schema_version": "1",
                }
            },
        }
    )
    assert DataSetManifest.model_validate_json(m.model_dump_json()) == m
    assert table_filename("patrol_legs") == "data_sets/patrol_legs.arrow"


def test_validate_table_names_mistyped_fields(spatial_feature_groups_table):
    idx = spatial_feature_groups_table.schema.get_field_index("pk")
    bad = spatial_feature_groups_table.set_column(idx, "pk", pa.array([1, 2]))
    with pytest.raises(DataSetSchemaError, match=r"spatial_feature_groups.*mistyped.*pk"):
        validate_table(bad, "spatial_feature_groups")
