from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point

from ecoscope import Trajectory
from ecoscope.analysis import astronomy
from tests.conftest import ARCTIC, EQUATOR, Segment, build_segments

# A civil-twilight night window used by the calculate_night_fraction unit tests: dusk at
# 18:00 flowing into dawn at 06:00 the next morning (a single contiguous interval).
DUSK = datetime(2024, 1, 1, 18, 0)
DAWN = datetime(2024, 1, 2, 6, 0)


def test_to_EarthLocation(movebank_relocations):
    geometry = movebank_relocations.gdf["geometry"]
    test_point = geometry.iloc[0]

    transformed = astronomy.to_EarthLocation(geometry)

    assert len(geometry) == len(transformed)

    transform = pyproj.Transformer.from_proj(
        proj_from="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        proj_to="+proj=geocent +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
    )

    # Check the projected values in the returned EarthLocation are what we expect
    test_val = transform.transform(xx=test_point.x, yy=test_point.y, zz=0)
    assert test_val[0] == transformed[0].x.value
    assert test_val[1] == transformed[0].y.value
    assert test_val[2] == transformed[0].z.value


def test_is_night(movebank_relocations):
    subset = movebank_relocations.gdf.iloc[12:15].copy()

    subset["is_night"] = astronomy.is_night(subset.geometry, subset.fixtime)

    assert subset["is_night"].values.tolist() == [True, True, False]


@pytest.mark.parametrize(
    "timezone",
    [
        timezone(timedelta(hours=10)),
        timezone.utc,
        timezone(timedelta(hours=-6)),
    ],
)
def test_nightday_ratio_salif_habiba(movebank_relocations, timezone):
    # movebank_relocations is subsampled to keep execution speed low.
    movebank_relocations.gdf = movebank_relocations.gdf.groupby("groupby_col", group_keys=False).head(100)

    trajectory = Trajectory.from_relocations(movebank_relocations)
    expected = pd.Series(
        [EXPECTED_HABIBA, EXPECTED_SALIF],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    # test against a handful of timezones to ensure this calculation is agnostic of input timezone
    trajectory.gdf["segment_start"] = trajectory.gdf["segment_start"].dt.tz_convert(timezone).dt.as_unit("ns")
    pd.testing.assert_series_equal(
        trajectory.gdf.groupby("groupby_col")[trajectory.gdf.columns].apply(
            lambda g: astronomy.get_nightday_ratio(g).ratio, include_groups=False
        ),
        expected,
    )


# Regenerated for the speed-based estimator (see get_nightday_ratio).
EXPECTED_HABIBA = 0.30657434444949494
EXPECTED_SALIF = 1.41640270989187


# --- get_nightday_ratio: synthetic tracks ------------------------------------------------
#
# Segments are pinned to a *local* solar clock so tests are agnostic of longitude/timezone.
# A solar unit runs noon-to-noon with its (contiguous) night in the middle; local hours are
# measured from local midnight 2024-03-20, so hours in [12, 36) fall in the single unit
# centred on local midnight 2024-03-21 (equator dusk ~18:31, dawn ~05:43). Every segment is
# kept at or below the default 6h gap threshold unless a test is exercising that drop.

_BASE = pd.Timestamp("2024-03-20 00:00")  # naive local midnight starting the reference day


def local_segments(point, specs, lon=0.0, bearing_deg=45.0):
    """Build a segment GeoDataFrame from (local_start_hour, local_end_hour, dist_meters) specs.

    Hours are measured from local midnight 2024-03-20; the UTC timestamps are back-computed
    from ``lon`` so the local placement -- and hence the day/night classification -- is
    identical at every longitude.
    """
    tz_offset = pd.Timedelta(hours=lon / 15.0)
    return build_segments(
        [
            Segment(
                point,
                (_BASE + pd.Timedelta(hours=h0) - tz_offset).tz_localize("UTC"),
                (_BASE + pd.Timedelta(hours=h1) - tz_offset).tz_localize("UTC"),
                dist,
                bearing_deg=bearing_deg,
            )
            for h0, h1, dist in specs
        ]
    )


@pytest.mark.parametrize("lon", [0.0, 60.0, 120.0, -60.0, -120.0, 23.0, -77.0])
def test_nightday_ratio_balanced_speed_is_one(lon):
    # Equal speed in one clear-day and one clear-night window of the same solar unit -> ratio
    # 1.0 at any longitude (the UTC placement shifts with lon, the local placement does not).
    gdf = local_segments(Point(lon, 0.0), [(14, 16, 2000.0), (20, 22, 2000.0)], lon=lon)
    result = astronomy.get_nightday_ratio(gdf)
    assert result.ratio == pytest.approx(1.0, rel=1e-6)
    assert result.n_days == 1


@pytest.mark.parametrize("lon", [0.0, 150.0, -150.0, 179.0])
def test_nightday_ratio_faster_by_day_is_diurnal(lon):
    # 3x the speed by day as by night (equal hours, 3:1 distance) -> night share 0.25 -> ratio
    # 1/3, at extreme longitudes too. Regression: local daytime falls at UTC times that "look
    # like" night, and must not be misclassified.
    gdf = local_segments(Point(lon, 0.0), [(14, 16, 3000.0), (20, 22, 1000.0)], lon=lon)
    assert astronomy.get_nightday_ratio(gdf).ratio == pytest.approx(1.0 / 3.0, rel=1e-6)


@pytest.mark.parametrize("bearing_deg", [0.0, 90.0, 180.0, 45.0], ids=["north", "east", "south", "north-east"])
def test_nightday_ratio_invariant_to_segment_direction(bearing_deg):
    # The ratio uses only the start point (solar offset + night window), dist_meters, and time,
    # never the segment's bearing, so movement in any direction gives the same answer.
    gdf = local_segments(EQUATOR, [(14, 16, 3000.0), (20, 22, 1000.0)], bearing_deg=bearing_deg)
    assert astronomy.get_nightday_ratio(gdf).ratio == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_nightday_ratio_speed_normalization_worked_example():
    # 4800 m over 8 observed daylight hours (600 m/h) and 4000 m over 4 night hours (1000 m/h):
    # night share = 1000 / (1000 + 600) = 0.625 -> ratio 1.667. A distance-only estimator would
    # call this diurnal (4000 / 8800 = 0.45); normalising by observed hours does not.
    gdf = local_segments(
        EQUATOR,
        [
            (13, 15, 1200.0),
            (15, 17, 1200.0),  # afternoon daylight, 4h / 2400 m
            (30, 32, 1200.0),
            (32, 34, 1200.0),  # morning daylight, 4h / 2400 m
            (20, 22, 2000.0),
            (22, 24, 2000.0),  # night, 4h / 4000 m
        ],
    )
    result = astronomy.get_nightday_ratio(gdf)
    assert result.ratio == pytest.approx(0.625 / (1 - 0.625), rel=1e-6)
    assert result.n_days == 1


@pytest.mark.parametrize("date", ["2024-03-20", "2024-12-21"], ids=["equinox", "winter-long-night"])
def test_nightday_ratio_cathemeral_invariant_to_night_length(date):
    # Same speed by day and by night -> ratio ~1.0 whether the night is ~12h (equinox) or ~15h
    # (winter) at lat 55. This is exactly the night-length bias the speed estimator removes: a
    # distance estimator would read diurnal in short-night seasons and nocturnal in long ones.
    base = pd.Timestamp(date + " 00:00")

    def seg(h0, h1, dist):
        return Segment(
            Point(0.0, 55.0),
            (base + pd.Timedelta(hours=h0)).tz_localize("UTC"),
            (base + pd.Timedelta(hours=h1)).tz_localize("UTC"),
            dist,
        )

    # 1000 m/h in both phases; 13:00-15:00 is daylight and 20:00-22:00 is night at lat 55 on
    # both dates, and both lie in the unit centred on the following midnight.
    gdf = build_segments([seg(13, 15, 2000.0), seg(20, 22, 2000.0)])
    result = astronomy.get_nightday_ratio(gdf)
    assert result.ratio == pytest.approx(1.0, rel=1e-6)
    assert result.n_days == 1


def test_nightday_ratio_night_across_midnight_is_one_unit():
    # A night bout straddling local midnight (22:00 -> 02:00) belongs to a single noon-to-noon
    # solar unit, not split across two days. With a daytime counterpart it forms one measurable
    # unit whose whole night is scored together.
    gdf = local_segments(
        EQUATOR,
        [
            (14, 16, 2000.0),  # day, 2h / 2000 m -> 1000 m/h
            (22, 24, 2000.0),  # night before midnight
            (24, 26, 2000.0),  # night after midnight (same unit)
        ],
    )
    result = astronomy.get_nightday_ratio(gdf)
    assert result.n_days == 1
    # night 4000 m / 4h = 1000 m/h equals the day speed -> balanced.
    assert result.ratio == pytest.approx(1.0, rel=1e-6)


def test_nightday_ratio_reports_contributing_unit_count():
    # Three consecutive solar units, each with balanced day+night movement -> n_days == 3.
    gdf = local_segments(
        EQUATOR,
        [
            (14, 16, 2000.0),
            (20, 22, 2000.0),  # unit centred 03-21
            (38, 40, 2000.0),
            (44, 46, 2000.0),  # +24h: unit centred 03-22
            (62, 64, 2000.0),
            (68, 70, 2000.0),  # +48h: unit centred 03-23
        ],
    )
    result = astronomy.get_nightday_ratio(gdf)
    assert result.n_days == 3
    assert result.ratio == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("lat", [-33.0, -10.0, 45.0])
def test_nightday_ratio_symmetric_across_latitudes(lat):
    # Balanced day/night speed -> ratio 1.0 at any non-polar latitude, in either hemisphere.
    gdf = local_segments(Point(0.0, lat), [(14, 16, 2000.0), (20, 22, 2000.0)])
    result = astronomy.get_nightday_ratio(gdf)
    assert result.ratio == pytest.approx(1.0, rel=1e-6)
    assert result.n_days == 1


def test_nightday_ratio_gates_units_with_thin_phase_coverage():
    # Unit A has >1h observed in both phases and contributes; unit B has only 0.5h of night, so
    # it fails the coverage gate and is set aside rather than dividing a distance by ~0 time.
    gdf = local_segments(
        EQUATOR,
        [
            (14, 16, 2000.0),
            (20, 22, 2000.0),  # unit centred 03-21: fully observed
            (38, 40, 2000.0),
            (44.0, 44.5, 500.0),  # unit centred 03-22: only 0.5h of night
        ],
    )
    result = astronomy.get_nightday_ratio(gdf)
    assert result.n_days == 1
    assert result.ratio == pytest.approx(1.0, rel=1e-6)


def test_nightday_ratio_weights_units_by_coverage():
    # Unit1 is well observed and diurnal (night share 0.25, 8h coverage); unit2 is a thinner
    # nocturnal unit (share 0.75, 4h coverage). Coverage-weighting pulls the mean toward unit1,
    # so the result is diurnal -- an unweighted mean would sit at 0.5 (ratio 1.0).
    gdf = local_segments(
        EQUATOR,
        [
            # unit centred 03-21: day 6000 m / 4h = 1500 m/h, night 2000 m / 4h = 500 m/h -> .25
            (14, 16, 3000.0),
            (16, 18, 3000.0),
            (20, 22, 1000.0),
            (22, 24, 1000.0),
            # unit centred 03-22: day 1000 m / 2h = 500 m/h, night 3000 m / 2h = 1500 m/h -> .75
            (38, 40, 1000.0),
            (44, 46, 3000.0),
        ],
    )
    result = astronomy.get_nightday_ratio(gdf)
    assert result.n_days == 2
    mean_share = (0.25 * 8 + 0.75 * 4) / 12
    assert result.ratio == pytest.approx(mean_share / (1 - mean_share), rel=1e-6)


def test_nightday_ratio_zero_daytime_movement_is_inf():
    # Daytime was observed (>1h) but the animal did not move then (day speed 0); it did move at
    # night. The night share is 1.0 -> an infinite (fully nocturnal) ratio, unit still counted.
    gdf = local_segments(EQUATOR, [(14, 16, 0.0), (20, 22, 2000.0)])
    result = astronomy.get_nightday_ratio(gdf)
    assert np.isinf(result.ratio)
    assert result.n_days == 1


def test_nightday_ratio_night_only_track_is_nan():
    # Movement only ever at night, with no daytime observation at all: every unit fails the day
    # coverage gate, so nothing is measurable -> NaN, n_days 0. (Contrast the old behaviour,
    # which kept such days as night-fraction 1.0 and could report a huge finite ratio.)
    gdf = local_segments(EQUATOR, [(20, 22, 2000.0), (44, 46, 2000.0)])
    result = astronomy.get_nightday_ratio(gdf)
    assert np.isnan(result.ratio)
    assert result.n_days == 0


def test_nightday_ratio_drops_long_gap_segments():
    # A single long gap-bridging segment (>6h default) is dropped, not apportioned by elapsed
    # time. Here it is the only segment, so the result degrades to NaN rather than inventing a
    # constant-speed day/night split for a chord the animal never actually traced.
    gdf = local_segments(EQUATOR, [(13, 21, 8000.0)])  # 8h segment spanning dusk
    result = astronomy.get_nightday_ratio(gdf)
    assert np.isnan(result.ratio)
    assert result.n_days == 0


def test_nightday_ratio_gap_threshold_is_configurable():
    # The 8h night segment is dropped by default (leaving only a day segment -> night gated ->
    # NaN) but kept once the threshold is relaxed, giving a balanced ratio.
    gdf = local_segments(EQUATOR, [(14, 16, 2000.0), (19, 27, 8000.0)])
    assert np.isnan(astronomy.get_nightday_ratio(gdf).ratio)

    kept = astronomy.get_nightday_ratio(gdf, max_segment_gap_hours=10.0)
    assert kept.ratio == pytest.approx(1.0, rel=1e-6)  # night 8000/8h == day 2000/2h == 1000 m/h
    assert kept.n_days == 1


def test_nightday_ratio_civil_twilight_bright_night_is_not_night():
    # At lat 65 near midsummer the sun never drops below -6 deg, so there is no civil night: a
    # "night-time" excursion is really twilight. night_window yields NaT, the unit is set aside
    # -> n_days 0. (With a horizon of 0 deg astroplan would still report a brief night here.)
    base = pd.Timestamp("2024-06-21 00:00")

    def seg(h0, h1, dist):
        return Segment(
            Point(0.0, 65.0),
            (base + pd.Timedelta(hours=h0)).tz_localize("UTC"),
            (base + pd.Timedelta(hours=h1)).tz_localize("UTC"),
            dist,
        )

    gdf = build_segments([seg(13, 15, 2000.0), seg(23, 25, 2000.0)])
    result = astronomy.get_nightday_ratio(gdf)
    assert result.n_days == 0
    assert np.isnan(result.ratio)


@pytest.mark.parametrize("date", [datetime(2024, 6, 21), datetime(2024, 12, 21)])
def test_nightday_ratio_polar_returns_nan(date):
    # At lat 80 the sun never sets (June) / never rises (December): night_window yields NaT,
    # every unit drops out, and the ratio degrades to NaN (n_days 0) rather than crashing.
    d = pd.Timestamp(date, tz="UTC")
    gdf = build_segments(
        [Segment(ARCTIC, d + pd.Timedelta(hours=h), d + pd.Timedelta(hours=h + 1)) for h in [2, 6, 12, 18]]
    )
    result = astronomy.get_nightday_ratio(gdf)
    assert np.isnan(result.ratio)
    assert result.n_days == 0


# --- calculate_night_fraction -------------------------------------------------------------


def _night_fraction_one(dusk, dawn, segment_start, segment_end):
    result = astronomy.calculate_night_fraction(
        dusk=pd.Series([dusk]),
        dawn=pd.Series([dawn]),
        segment_start=pd.Series([segment_start]),
        segment_end=pd.Series([segment_end]),
    )
    return result[0]


@pytest.mark.parametrize(
    "segment_start, segment_end, expected, label",
    [
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12), 0.0, "all day before dusk"),
        (datetime(2024, 1, 1, 20), datetime(2024, 1, 1, 22), 1.0, "all night before midnight"),
        (datetime(2024, 1, 2, 2), datetime(2024, 1, 2, 4), 1.0, "all night after midnight"),
        (datetime(2024, 1, 2, 8), datetime(2024, 1, 2, 10), 0.0, "all day after dawn"),
        (datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 19), 0.5, "day -> night across dusk"),
        (datetime(2024, 1, 2, 5), datetime(2024, 1, 2, 7), 0.5, "night -> day across dawn"),
        (datetime(2024, 1, 1, 12), datetime(2024, 1, 2, 12), 0.5, "full 24h contains the whole night"),
        # boundaries: touching dusk/dawn exactly assigns to the adjacent open interval.
        (datetime(2024, 1, 1, 18), datetime(2024, 1, 1, 20), 1.0, "start exactly at dusk"),
        (datetime(2024, 1, 2, 4), datetime(2024, 1, 2, 6), 1.0, "end exactly at dawn"),
        (datetime(2024, 1, 1, 16), datetime(2024, 1, 1, 18), 0.0, "end exactly at dusk"),
        (datetime(2024, 1, 2, 6), datetime(2024, 1, 2, 8), 0.0, "start exactly at dawn"),
    ],
)
def test_calculate_night_fraction(segment_start, segment_end, expected, label):
    actual = _night_fraction_one(DUSK, DAWN, segment_start, segment_end)
    assert actual == pytest.approx(expected), f"{label}: got {actual}, expected {expected}"


def test_calculate_night_fraction_nat_window_is_nan():
    # A unit with no qualifying darkness (NaT dusk/dawn) yields NaN, which the caller drops.
    assert np.isnan(_night_fraction_one(pd.NaT, pd.NaT, datetime(2024, 1, 1, 20), datetime(2024, 1, 1, 22)))
    assert np.isnan(_night_fraction_one(DUSK, pd.NaT, datetime(2024, 1, 1, 20), datetime(2024, 1, 1, 22)))


def test_calculate_night_fraction_vectorized():
    """Run several rows in one call to confirm vectorization preserves alignment."""
    rows = [
        (datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 19), 0.5),
        (datetime(2024, 1, 1, 20), datetime(2024, 1, 1, 22), 1.0),
        (datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 12), 0.0),
        (datetime(2024, 1, 2, 5), datetime(2024, 1, 2, 7), 0.5),
    ]
    starts, ends, expected = zip(*rows)
    actual = astronomy.calculate_night_fraction(
        dusk=pd.Series([DUSK] * len(rows)),
        dawn=pd.Series([DAWN] * len(rows)),
        segment_start=pd.Series(starts),
        segment_end=pd.Series(ends),
    )
    np.testing.assert_allclose(actual, expected)


def test_night_window_unresolved_returns_nat():
    # At a polar-day latitude the sun never crosses -6 deg within astroplan's search window, so
    # astroplan returns a masked 0-d value. night_window must coerce it to NaT rather than
    # passing the 0-d array through, which would later crash get_nightday_ratio with
    # "iteration over a 0-d array" during the calculate_night_fraction comparisons.
    result = astronomy.night_window(datetime(2025, 6, 21), Point(0.0, 80.0))
    assert pd.isna(result["dusk"])
    assert pd.isna(result["dawn"])


def test_night_window_is_a_contiguous_utc_interval():
    # A normal night: dusk precedes dawn, both are tz-aware UTC, and the window brackets the
    # local midnight of the requested unit.
    result = astronomy.night_window(datetime(2024, 3, 21), Point(0.0, 0.0))
    assert result["dusk"] < result["dawn"]
    assert result["dusk"].tzinfo is not None and result["dawn"].tzinfo is not None


# --- calculate_day_night_distance (legacy, scalar helper -- unchanged) --------------------


@pytest.fixture
def daily_summary():
    """Create a sample daily summary DataFrame for testing."""
    date = datetime(2024, 1, 1)
    df = pd.DataFrame(
        {
            "sunrise": [datetime(2024, 1, 1, 6, 0)],
            "sunset": [datetime(2024, 1, 1, 18, 0)],
            "day_distance": [0.0],
            "night_distance": [0.0],
        },
        index=[date],
    )
    return df


def test_all_night_before_sunrise(daily_summary):
    """Test segment entirely before sunrise."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 2, 0), datetime(2024, 1, 1, 4, 0), 1000, daily_summary
    )  # 2:00 AM  # 4:00 AM
    assert daily_summary.loc[date, "night_distance"] == 1000
    assert daily_summary.loc[date, "day_distance"] == 0


def test_all_night_after_sunset(daily_summary):
    """Test segment entirely after sunset."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 20, 0), datetime(2024, 1, 1, 22, 0), 1000, daily_summary
    )  # 8:00 PM  # 10:00 PM
    assert daily_summary.loc[date, "night_distance"] == 1000
    assert daily_summary.loc[date, "day_distance"] == 0


def test_all_day(daily_summary):
    """Test segment entirely during daylight hours."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 14, 0), 1000, daily_summary
    )  # 10:00 AM  # 2:00 PM
    assert daily_summary.loc[date, "day_distance"] == 1000
    assert daily_summary.loc[date, "night_distance"] == 0


def test_day_to_night_transition(daily_summary):
    """Test segment starting in day and ending in night."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 17, 0), datetime(2024, 1, 1, 19, 0), 1000, daily_summary
    )  # 5:00 PM  # 7:00 PM
    # Segment spans 2 hours, with 1 hour in day and 1 hour in night
    assert daily_summary.loc[date, "day_distance"] == 500
    assert daily_summary.loc[date, "night_distance"] == 500


def test_night_to_day_transition(daily_summary):
    """Test segment starting in night and ending in day."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 5, 0), datetime(2024, 1, 1, 7, 0), 1000, daily_summary
    )  # 5:00 AM  # 7:00 AM
    # Segment spans 2 hours, with 1 hour in night and 1 hour in day
    assert daily_summary.loc[date, "day_distance"] == 500
    assert daily_summary.loc[date, "night_distance"] == 500


@pytest.fixture
def inverted_daily_summary():
    """Create a daily summary where sunrise is after sunset in UTC."""
    date = datetime(2024, 1, 1)
    df = pd.DataFrame(
        {
            "sunrise": [datetime(2024, 1, 1, 18, 0)],  # 6:00 PM
            "sunset": [datetime(2024, 1, 1, 6, 0)],  # 6:00 AM
            "day_distance": [0.0],
            "night_distance": [0.0],
        },
        index=[date],
    )
    return df


def test_inverted_all_day(inverted_daily_summary):
    """Test all-day segment when sunrise is after sunset."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date,
        datetime(2024, 1, 1, 4, 0),
        datetime(2024, 1, 1, 5, 0),
        1000,
        inverted_daily_summary,  # 4:00 AM  # 5:00 AM
    )
    assert inverted_daily_summary.loc[date, "day_distance"] == 1000
    assert inverted_daily_summary.loc[date, "night_distance"] == 0


def test_inverted_all_night(inverted_daily_summary):
    """Test all-night segment when sunrise is after sunset."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date,
        datetime(2024, 1, 1, 7, 0),
        datetime(2024, 1, 1, 17, 0),
        1000,
        inverted_daily_summary,  # 7:00 AM  # 5:00 PM
    )
    assert inverted_daily_summary.loc[date, "night_distance"] == 1000
    assert inverted_daily_summary.loc[date, "day_distance"] == 0
