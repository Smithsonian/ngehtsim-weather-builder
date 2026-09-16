from datetime import date
import json

import numpy as np
import pytest

from ngehtsim_weather_builder.legacy import (
    LegacyFormatError,
    WeatherPartition,
    WeatherRecords,
    normalize_legacy_partition,
    read_legacy_partition,
    write_legacy_partition,
)
from ngehtsim_weather_builder.normalize import _normalization_lock, main


def _partition(year_month_days, component_count=2):
    daily_year = np.array([year for year, _, _ in year_month_days], dtype="<i2")
    daily_month = np.array([month for _, month, _ in year_month_days], dtype="i1")
    daily_day = np.array([day for _, _, day in year_month_days], dtype="i1")
    daily_count = len(daily_year)
    daily_index = np.arange(daily_count, dtype=np.float64)
    daily = WeatherRecords(
        year=daily_year,
        month=daily_month,
        day=daily_day,
        time_index=None,
        tau_coefficients=np.full((daily_count, component_count), 1.0, dtype=np.float16),
        tb_coefficients=np.full((daily_count, component_count), 2.0, dtype=np.float16),
        pwv_mm=daily_index + 3.0,
        wind_speed_m_s=daily_index + 4.0,
        surface_pressure_mbar=daily_index + 5.0,
        surface_temperature_k=daily_index + 6.0,
    )
    native_count = daily_count * 8
    native = WeatherRecords(
        year=np.repeat(daily_year, 8),
        month=np.repeat(daily_month, 8),
        day=np.repeat(daily_day, 8),
        time_index=np.tile(np.arange(8, dtype="i1"), daily_count),
        tau_coefficients=np.full((native_count, component_count), 1.0, dtype=np.float16),
        tb_coefficients=np.full((native_count, component_count), 2.0, dtype=np.float16),
        pwv_mm=np.arange(native_count, dtype=np.float64) + 3.0,
        wind_speed_m_s=np.arange(native_count, dtype=np.float64) + 4.0,
        surface_pressure_mbar=np.arange(native_count, dtype=np.float64) + 5.0,
        surface_temperature_k=np.arange(native_count, dtype=np.float64) + 6.0,
    )
    return WeatherPartition(native=native, daily=daily)


def _append_daily_row(directory, *, finite):
    for name, atmospheric in (
        ("tau", True),
        ("Tb", True),
        ("PWV", False),
        ("windspeed", False),
        ("Pbase", False),
        ("Tbase", False),
    ):
        path = directory / (name + ".txt")
        contents = path.read_bytes()
        record_size = int.from_bytes(contents[:2], "little")
        row = bytearray(contents[-record_size:])
        if atmospheric and not finite:
            row[-2:] = np.float16(np.nan).tobytes()
        path.write_bytes(contents + bytes(row))


def test_normalizer_removes_invalid_duplicate_daily_row(tmp_path):
    source = tmp_path / "04Apr"
    write_legacy_partition(source, _partition([(2026, 4, 30)]), component_count=2)
    _append_daily_row(source, finite=False)

    result = normalize_legacy_partition(source, date(2026, 4, 30), component_count=2)

    assert result.removed_native_records == 0
    assert result.removed_daily_records == 1
    assert result.partition.daily.count == 1
    np.testing.assert_array_equal(result.partition.daily.day, [30])


def test_normalizer_rejects_ambiguous_finite_daily_duplicate(tmp_path):
    source = tmp_path / "04Apr"
    write_legacy_partition(source, _partition([(2026, 4, 30)]), component_count=2)
    _append_daily_row(source, finite=True)

    with pytest.raises(LegacyFormatError, match="multiple finite rows"):
        normalize_legacy_partition(source, date(2026, 4, 30), component_count=2)


def test_normalizer_drops_native_records_after_cutoff(tmp_path):
    source = tmp_path / "08Aug"
    write_legacy_partition(
        source,
        _partition([(2025, 8, 31), (2026, 8, 1)]),
        component_count=2,
    )

    result = normalize_legacy_partition(source, date(2026, 7, 31), component_count=2)

    assert result.removed_native_records == 8
    assert result.removed_daily_records == 1
    assert result.partition.native.count == 8
    assert result.partition.daily.count == 1
    np.testing.assert_array_equal(result.partition.daily.year, [2025])


def test_normalizer_preserves_retained_binary_records_byte_for_byte(tmp_path):
    source = tmp_path / "04Apr"
    destination = tmp_path / "normalized"
    write_legacy_partition(
        source,
        _partition([(2025, 4, 30), (2026, 4, 30)]),
        component_count=2,
    )

    result = normalize_legacy_partition(source, date(2026, 4, 30), component_count=2)
    write_legacy_partition(destination, result.partition, component_count=2)

    for source_file in sorted(source.glob("*.txt")):
        assert source_file.read_bytes() == (destination / source_file.name).read_bytes()


def test_normalization_lock_rejects_concurrent_destination(tmp_path):
    daily_output = tmp_path / "weather_data"

    with _normalization_lock(daily_output):
        lock_path = tmp_path / ".weather_data.normalization.lock"
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="already using"):
            with _normalization_lock(daily_output):
                pass

    assert not lock_path.exists()


def _month_days(year, month):
    final_day = 28 if month == 2 else (30 if month in (4, 6, 9, 11) else 31)
    return [(year, month, day) for day in range(1, final_day + 1)]


def test_normalizer_cli_writes_matching_bounded_archives(tmp_path):
    source_root = tmp_path / "source"
    for month in range(1, 13):
        dates = _month_days(2025, month)
        if month <= 7:
            dates += _month_days(2026, month)
        elif month == 8:
            dates += [(2026, 8, 1)]
        label = "{0:02d}".format(month) + date(2000, month, 1).strftime("%b")
        write_legacy_partition(
            source_root / "TEST" / label,
            _partition(dates),
            component_count=2,
        )

    registry = tmp_path / "sites.csv"
    registry.write_text("Name\nTEST\n", encoding="utf-8")
    daily_output = tmp_path / "daily"
    alltimes_output = tmp_path / "alltimes"
    report = tmp_path / "report.json"

    assert main([
        "--input-root", str(source_root),
        "--daily-output", str(daily_output),
        "--alltimes-output", str(alltimes_output),
        "--site-registry", str(registry),
        "--cutoff-date", "2026-07-31",
        "--report", str(report),
        "--component-count", "2",
        "--start-year", "2025",
    ]) == 0

    august = read_legacy_partition(alltimes_output / "TEST" / "08Aug", component_count=2)
    np.testing.assert_array_equal(august.daily.year, [2025] * 31)
    assert (daily_output / "TEST" / "08Aug" / "PWV.txt").read_bytes() == (
        alltimes_output / "TEST" / "08Aug" / "PWV.txt"
    ).read_bytes()
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["validation"]["status"] == "passed"
    assert result["totals"]["removed_native_records"] == 8
    assert result["totals"]["removed_daily_records"] == 1
