"""Shared fixtures for the platform task tests.

The warehouse observations table factory lives here rather than in the individual
warehouse test modules: it is built against ``OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1``,
and ``pa.table(..., schema=...)`` raises ``KeyError`` for any schema field it is not
given an array for. That broke every copy of this helper when v0.0.22 added
``subject_additional`` and again when v0.0.28 added the promoted subject fields, so
the factory now null-fills any schema field it does not set explicitly.
"""

from datetime import datetime, timezone

import pytest


@pytest.fixture
def warehouse_observations_table():
    """Callable building a ``pa.Table`` shaped like ``ERWarehouseClient`` observations.

    ``fixtimes`` are emitted in the order given, which is what lets callers simulate the
    warehouse returning rows in an arbitrary order; omit them for one fix per subject a
    minute apart. ``additional`` is one value per row for ``extra__subject__additional``
    -- pass None for an all-null column, which is what the warehouse serves when the
    caller does not set ``include_subject_additional``. ``sex`` likewise fills
    ``extra__subject__sex``; the other promoted subject fields are always null.
    """

    def _make(subject_ids=("s1", "s2"), fixtimes=None, sources=None, additional=None, sex=None):
        import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
        import pyarrow as pa
        from ecoscope_earthranger_io_core.arrow import (
            OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1,
        )
        from shapely.geometry import Point

        subject_ids = list(subject_ids)
        n = len(subject_ids)
        if fixtimes is None:
            fixtimes = [datetime(2015, 1, 1, minute=i, tzinfo=timezone.utc) for i in range(n)]
        if sources is None:
            sources = [f"source-{s}" for s in subject_ids]

        columns = {
            "geometry": ga.array([Point(37.5 + i * 0.001, -2.5).wkb for i in range(n)]),
            "fixtime": pa.array(list(fixtimes), type=pa.timestamp("ns", tz="UTC")),
            "groupby_col": pa.array(subject_ids, type=pa.string()),
            # Duplicates groupby_col, matching what the warehouse serves: io-core
            # v0.0.23 added this so the subject id survives under the name the
            # EarthRanger API path uses (`subject__id`, once `extra__` is stripped).
            "extra__subject__id": pa.array(subject_ids, type=pa.string()),
            "extra__subject__name": pa.array([f"subj-{s}" for s in subject_ids], type=pa.string()),
            "extra__subject__subject_subtype": pa.array(["elephant"] * n, type=pa.string()),
            "extra__subject__additional": pa.array(
                list(additional) if additional is not None else [None] * n,
                type=pa.string(),
            ),
            "extra__source": pa.array(list(sources), type=pa.string()),
            "junk_status": pa.array([False] * n, type=pa.bool_()),
            "extra__subject__sex": pa.array(list(sex) if sex is not None else [None] * n, type=pa.string()),
        }
        for field in OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1:
            columns.setdefault(field.name, pa.nulls(n, type=field.type))
        return pa.table(columns, schema=OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1)

    return _make
