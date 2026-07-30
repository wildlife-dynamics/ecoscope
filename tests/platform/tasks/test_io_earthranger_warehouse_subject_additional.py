"""Unit tests for the subject ``additional`` opt-in on the ERWarehouseClient
observations path (ERDW-247; consumes the warehouse-side flag added in ERDW-266).

The EarthRanger API path serves the subject's ``additional`` JSON unconditionally, as
part of ``include_subject_details``. The warehouse serves the column but leaves it null
unless the query asks for it, so a consumer that reads ``additional`` -- e.g.
``assign_subject_colors``, which colors tracks from its ``rgb`` key -- gets different
results from the two backends unless the task opts in. These tests pin the flag's
forwarding and the resulting cross-backend parity.

They use a mocked ERWarehouseClient (no live server), so unlike the integration tests in
test_io_earthranger.py they are NOT marked ``io`` and run in the default test job.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from pydantic import TypeAdapter
from shapely.geometry import Point

from ecoscope.platform.annotations import EmptyDataFrame
from ecoscope.platform.schemas import SubjectGroupObservationsGDF
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import get_subjectgroup_observations
from ecoscope.platform.tasks.transformation import assign_subject_colors, drop_column_prefix

# The task's declared return type, validated in production: wt_task wraps every task in
# `validate_call(..., validate_return=True)`. Asserting through this rather than through
# `StrictSubjectGroupObservationsGDFSchema` directly also exercises the AfterValidators
# that inject the other optional columns.
_RETURN_TYPE = TypeAdapter(SubjectGroupObservationsGDF | EmptyDataFrame)

_TIME_RANGE = TimeRange(
    since=datetime(2015, 1, 1, tzinfo=timezone.utc),
    until=datetime(2015, 1, 31, tzinfo=timezone.utc),
    timezone=UTC_TIMEZONEINFO,
)

# The format EarthRanger stores in `additional`: comma-separated decimal channels, not
# hex (see `parse_rgb_str` in tasks/transformation/_subjects.py).
_RED = json.dumps({"rgb": "255, 0, 0", "sex": "female", "region": "Gourma"})
_BLUE = json.dumps({"rgb": "0, 0, 255", "sex": "male", "region": "Gourma"})


def _call_via_warehouse(table, **task_kwargs):
    """Run the task against a mocked warehouse client; return (result, call kwargs)."""
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_subjectgroup_observations.return_value = table

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_subjectgroup_observations(
            client=MagicMock(),
            time_range=_TIME_RANGE,
            subject_group_name="Ecoscope-5-Subs",
            raise_on_empty=False,
            **task_kwargs,
        )

    return result, mock_warehouse_client.get_subjectgroup_observations.call_args.kwargs


class TestFlagForwarding:
    def test_defaults_to_not_requesting(self, warehouse_observations_table):
        """Opt-in: workflows that never read `additional` must not pay for it."""
        _, call_kwargs = _call_via_warehouse(warehouse_observations_table())

        assert call_kwargs["include_subject_additional"] is False

    def test_forwarded_when_requested(self, warehouse_observations_table):
        _, call_kwargs = _call_via_warehouse(
            warehouse_observations_table(additional=[_RED, _BLUE]),
            include_subject_additional=True,
        )

        assert call_kwargs["include_subject_additional"] is True

    @pytest.mark.parametrize("requested", [False, True])
    def test_not_forwarded_to_earthranger_api_client(self, requested):
        """The legacy client has no such kwarg -- forwarding it would be a TypeError.

        The EarthRanger API serves `additional` unconditionally, so there is nothing to
        request on that path.
        """
        mock_legacy_client = MagicMock()
        mock_legacy_client.get_subjectgroup_observations.return_value = gpd.GeoDataFrame()

        with patch(
            "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
            return_value=None,
        ):
            get_subjectgroup_observations(
                client=mock_legacy_client,
                time_range=_TIME_RANGE,
                subject_group_name="Ecoscope-5-Subs",
                raise_on_empty=False,
                include_subject_additional=requested,
            )

        call_kwargs = mock_legacy_client.get_subjectgroup_observations.call_args.kwargs
        assert "include_subject_additional" not in call_kwargs


class TestAdditionalColumnInResult:
    def test_json_reaches_the_dataframe(self, warehouse_observations_table):
        result, _ = _call_via_warehouse(
            warehouse_observations_table(additional=[_RED, _BLUE]),
            include_subject_additional=True,
        )

        assert json.loads(result.extra__subject__additional.iloc[0])["rgb"] == "255, 0, 0"
        assert json.loads(result.extra__subject__additional.iloc[1])["rgb"] == "0, 0, 255"

    def test_column_present_but_null_when_not_requested(self, warehouse_observations_table):
        """Null rather than absent: downstream column lookups stay stable either way."""
        result, _ = _call_via_warehouse(warehouse_observations_table())

        assert "extra__subject__additional" in result.columns
        assert result.extra__subject__additional.isna().all()

    @pytest.mark.parametrize(
        "additional",
        [
            pytest.param(None, id="all-null"),
            pytest.param([_RED, _BLUE], id="json"),
            pytest.param([_RED, None], id="mixed"),
            pytest.param(["{}", "{}"], id="empty-json"),
        ],
    )
    def test_result_satisfies_the_subject_group_schema(self, warehouse_observations_table, additional):
        """`extra__subject__additional` is declared optional and nullable, so each of
        the values the warehouse can serve -- null, '{}' or JSON -- validates."""
        result, _ = _call_via_warehouse(warehouse_observations_table(additional=additional))

        validated = _RETURN_TYPE.validate_python(result)

        # Nulls must survive validation as nulls. The column is deliberately left out of
        # `_subject_group_obs_optional_columns`, whose `fillna` would turn them into the
        # string "None" -- unparsable JSON, where null is already handled.
        if additional is None:
            assert validated.extra__subject__additional.isna().all()

    def test_earthranger_api_dict_additional_satisfies_the_schema(self):
        """The EarthRanger API serves `additional` as a dict, the warehouse as a JSON
        string, so the column cannot be typed `str`.

        `ecoscope/io/earthranger.py` reads it as a dict (`df["additional"].str["rgb"]`,
        and `pack_columns` splats it with `{**add_dict}`), and `assign_subject_colors`
        has an explicit `isinstance(additional_data, dict)` branch for it. Typing the
        column `str` would abort every subject-group workflow on a non-warehouse site.
        """
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1)],
                "groupby_col": ["s1", "s2"],
                "fixtime": pd.to_datetime(["2015-01-01", "2015-01-02"], utc=True),
                "junk_status": [False, False],
                "extra__subject__name": ["subj-s1", "subj-s2"],
                "extra__subject__subject_subtype": ["elephant"] * 2,
                "extra__subject__sex": ["female", "male"],
                "extra__subject__additional": [{"rgb": "255, 0, 0"}, {}],
            },
            crs=4326,
        )

        validated = _RETURN_TYPE.validate_python(gdf)

        assert validated.extra__subject__additional.iloc[0] == {"rgb": "255, 0, 0"}


class TestColoringParity:
    """The reason the flag exists: `assign_subject_colors` reading `rgb`.

    Mirrors the download-subject-tracks node chain -- warehouse observations, then
    `drop_column_prefix("extra__")`, then `assign_subject_colors` keyed on
    `groupby_col`.
    """

    def _colors(self, table, **task_kwargs):
        result, _ = _call_via_warehouse(table, **task_kwargs)
        result = drop_column_prefix(df=result, prefix="extra__")
        colored = assign_subject_colors(
            df=result,
            subject_id_column="groupby_col",
            additional_column="subject__additional",
            output_column="subject_colormap",
            fallback_strategy="default_color",
            default_color="#FFFF00",
        )
        return dict(zip(colored.groupby_col, colored.subject_colormap))

    def test_requested_additional_yields_each_subjects_own_color(self, warehouse_observations_table):
        colors = self._colors(
            warehouse_observations_table(additional=[_RED, _BLUE]),
            include_subject_additional=True,
        )

        assert colors["s1"] == (255, 0, 0, 255)
        assert colors["s2"] == (0, 0, 255, 255)

    @pytest.mark.parametrize(
        "additional",
        [
            pytest.param(None, id="not-requested-null"),
            pytest.param(["{}", "{}"], id="requested-no-attributes"),
        ],
    )
    def test_additional_without_rgb_falls_back_without_raising(self, warehouse_observations_table, additional):
        """Both no-rgb states must degrade to the default color, not crash: other
        workflows share this task and never opt in."""
        colors = self._colors(warehouse_observations_table(additional=additional))

        assert set(colors.values()) == {(255, 255, 0, 255)}

    def test_earthranger_api_dict_additional_yields_the_same_colors(self, warehouse_observations_table):
        """Cross-backend parity, which is the point of the flag: the ER API's dict and
        the warehouse's JSON string must color identically."""
        table = warehouse_observations_table(additional=[_RED, _BLUE])
        warehouse_colors = self._colors(table, include_subject_additional=True)

        er_api_frame = drop_column_prefix(
            df=gpd.GeoDataFrame(
                {
                    "groupby_col": ["s1", "s2"],
                    "extra__subject__additional": [json.loads(_RED), json.loads(_BLUE)],
                    "geometry": [Point(0, 0), Point(1, 1)],
                }
            ),
            prefix="extra__",
        )
        er_api_colors = dict(
            zip(
                er_api_frame.groupby_col,
                assign_subject_colors(
                    df=er_api_frame,
                    subject_id_column="groupby_col",
                    additional_column="subject__additional",
                    output_column="subject_colormap",
                    fallback_strategy="default_color",
                    default_color="#FFFF00",
                ).subject_colormap,
            )
        )

        assert er_api_colors == warehouse_colors == {"s1": (255, 0, 0, 255), "s2": (0, 0, 255, 255)}
