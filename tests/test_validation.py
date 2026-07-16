import json
from calendar import monthrange
from pathlib import Path
import struct

import numpy as np
import pytest
import zarr

from ngehtsim_weather_builder.cli import discover_partitions, main
from ngehtsim_weather_builder.dataset import PcaBasis, initialize_dataset, write_partition
from ngehtsim_weather_builder.legacy import WeatherPartition, WeatherRecords
from ngehtsim_weather_builder.validation import (
    DatasetValidationError,
    validate_complete_calendar_coverage,
    validate_written_partition,
)


def _basis():
    return PcaBasis(
        tau_mean=np.array([1.0, 2.0, 3.0, 4.0]),
        tau_components=np.arange(8, dtype=np.float64).reshape(2, 4) * 1e-3,
        tb_mean=np.array([5.0, 6.0, 7.0, 8.0]),
        tb_components=np.arange(8, 16, dtype=np.float64).reshape(2, 4) * 1e-3,
    )


def _complete_partition(year=2020, month=4, day_count=None):
    count_days = monthrange(year, month)[1] if day_count is None else day_count
    days = np.arange(1, count_days + 1, dtype="i1")
    daily_count = days.size
    native_count = daily_count * 8
    native_days = np.repeat(days, 8)
    return WeatherPartition(
        native=WeatherRecords(
            year=np.full(native_count, year, dtype="<i2"),
            month=np.full(native_count, month, dtype="i1"),
            day=native_days,
            time_index=np.tile(np.arange(8, dtype="i1"), daily_count),
            tau_coefficients=np.arange(native_count * 2, dtype=np.float16).reshape(native_count, 2),
            tb_coefficients=(
                np.arange(native_count * 2, dtype=np.float16).reshape(native_count, 2) + 100
            ),
            pwv_mm=np.arange(native_count, dtype=np.float64),
            wind_speed_m_s=np.arange(native_count, dtype=np.float64) + 1,
            surface_pressure_mbar=np.arange(native_count, dtype=np.float64) + 2,
            surface_temperature_k=np.arange(native_count, dtype=np.float64) + 3,
        ),
        daily=WeatherRecords(
            year=np.full(daily_count, year, dtype="<i2"),
            month=np.full(daily_count, month, dtype="i1"),
            day=days,
            time_index=None,
            tau_coefficients=np.arange(daily_count * 2, dtype=np.float16).reshape(daily_count, 2),
            tb_coefficients=(
                np.arange(daily_count * 2, dtype=np.float16).reshape(daily_count, 2) + 100
            ),
            pwv_mm=np.arange(daily_count, dtype=np.float64),
            wind_speed_m_s=np.arange(daily_count, dtype=np.float64) + 1,
            surface_pressure_mbar=np.arange(daily_count, dtype=np.float64) + 2,
            surface_temperature_k=np.arange(daily_count, dtype=np.float64) + 3,
        ),
    )


def _record_dtype(component_count, with_time, coefficients):
    fields = [("year", "<i2"), ("month", "i1"), ("day", "i1")]
    if with_time:
        fields.append(("time_index", "i1"))
    fields.append(("value", "<f2", (component_count,)) if coefficients else ("value", "<f8"))
    return np.dtype(fields)


def _write_legacy_records(path, records, component_count, with_time, coefficients, values):
    dtype = _record_dtype(component_count, with_time, coefficients)
    binary = np.zeros(records.count, dtype=dtype)
    binary["year"] = records.year
    binary["month"] = records.month
    binary["day"] = records.day
    if with_time:
        binary["time_index"] = records.time_index
    binary["value"] = values
    path.write_bytes(struct.pack("<H", dtype.itemsize) + binary.tobytes())


def _write_legacy_partition(directory, partition):
    directory.mkdir(parents=True)
    for suffix, records, with_time in (
        ("_alltimes", partition.native, True),
        ("", partition.daily, False),
    ):
        _write_legacy_records(
            directory / "tau{0}.txt".format(suffix),
            records,
            2,
            with_time,
            True,
            records.tau_coefficients,
        )
        _write_legacy_records(
            directory / "Tb{0}.txt".format(suffix),
            records,
            2,
            with_time,
            True,
            records.tb_coefficients,
        )
        for filename, values in (
            ("PWV", records.pwv_mm),
            ("windspeed", records.wind_speed_m_s),
            ("Pbase", records.surface_pressure_mbar),
            ("Tbase", records.surface_temperature_k),
        ):
            _write_legacy_records(
                directory / "{0}{1}.txt".format(filename, suffix),
                records,
                2,
                with_time,
                False,
                values,
            )


def _with_invalid_duplicate_daily_record(partition):
    daily = partition.daily
    duplicate_tau = np.concatenate((daily.tau_coefficients, daily.tau_coefficients[:1].copy()))
    duplicate_tau[-1] = np.nan
    duplicate_daily = WeatherRecords(
        year=np.concatenate((daily.year, daily.year[:1])),
        month=np.concatenate((daily.month, daily.month[:1])),
        day=np.concatenate((daily.day, daily.day[:1])),
        time_index=None,
        tau_coefficients=duplicate_tau,
        tb_coefficients=np.concatenate((daily.tb_coefficients, daily.tb_coefficients[:1])),
        pwv_mm=np.concatenate((daily.pwv_mm, daily.pwv_mm[:1])),
        wind_speed_m_s=np.concatenate((daily.wind_speed_m_s, daily.wind_speed_m_s[:1])),
        surface_pressure_mbar=np.concatenate(
            (daily.surface_pressure_mbar, daily.surface_pressure_mbar[:1])
        ),
        surface_temperature_k=np.concatenate(
            (daily.surface_temperature_k, daily.surface_temperature_k[:1])
        ),
    )
    return WeatherPartition(native=partition.native, daily=duplicate_daily)


def _write_basis(directory, basis):
    directory.mkdir()
    np.savetxt(directory / "spectrum_mean.txt", basis[0])
    for index, component in enumerate(basis[1]):
        np.savetxt(directory / "spectrum_{0:04d}.txt".format(index), component)


def test_validates_complete_calendar_coverage_and_exact_zarr_values(tmp_path):
    partition = _complete_partition()
    coverage = validate_complete_calendar_coverage(partition)
    assert coverage.daily_records == 30
    assert coverage.native_records == 240
    assert coverage.first_date == "2020-04-01"
    assert coverage.last_date == "2020-04-30"

    output = tmp_path / "weather.zarr"
    root = initialize_dataset(output, _basis(), np.arange(4, dtype=np.float64))
    write_partition(root, "ALMA", 4, partition)
    validate_written_partition(root, "ALMA", 4, partition)

    root["sites/ALMA/months/04/daily/pwv_mm"][0] = -1.0
    with pytest.raises(DatasetValidationError, match="do not match"):
        validate_written_partition(root, "ALMA", 4, partition)


def test_rejects_corrupted_native_summary_values(tmp_path):
    partition = _complete_partition()
    output = tmp_path / "weather.zarr"
    basis = _basis()
    root = initialize_dataset(output, basis, np.arange(4, dtype=np.float64))
    write_partition(root, "ALMA", 4, partition, basis=basis)

    root["sites/ALMA/months/04/native_summary/median/opacity"][0, 0] = -1.0
    with pytest.raises(DatasetValidationError, match="native summary values"):
        validate_written_partition(root, "ALMA", 4, partition, basis=basis)


def test_rejects_incomplete_calendar_month():
    with pytest.raises(DatasetValidationError, match="incomplete"):
        validate_complete_calendar_coverage(_complete_partition(day_count=29))


def test_cli_writes_manifest_and_discovers_all_partitions(tmp_path):
    partition = _complete_partition()
    legacy_root = tmp_path / "weather_data_alltimes"
    source = legacy_root / "ALMA" / "04Apr"
    _write_legacy_partition(source, partition)

    tau_basis = tmp_path / "tau_basis"
    tb_basis = tmp_path / "tb_basis"
    basis = _basis()
    _write_basis(tau_basis, (basis.tau_mean, basis.tau_components))
    _write_basis(tb_basis, (basis.tb_mean, basis.tb_components))
    site_registry = tmp_path / "Telescope_Site_Matrix.csv"
    site_registry.write_text("Station\nALMA\n", encoding="utf-8")

    discovered = discover_partitions(legacy_root)
    assert [(item.site, item.month, item.source) for item in discovered] == [("ALMA", 4, source)]

    output = tmp_path / "release.zarr"
    assert main(
        [
            "--output",
            str(output),
            "--dataset-id",
            "test-release-v0.1.0",
            "--builder-revision",
            "test-revision",
            "--legacy-root",
            str(legacy_root),
            "--site-registry",
            str(site_registry),
            "--tau-basis",
            str(tau_basis),
            "--tb-basis",
            str(tb_basis),
            "--component-count",
            "2",
            "--all-partitions",
        ]
    ) == 0

    manifest = json.loads((tmp_path / "release.zarr.manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["status"] == "passed"
    assert manifest["dataset"]["schema_version"] == "0.2.0"
    assert manifest["dataset"]["native_summary_forms"] == ["mean", "median", "good", "bad"]
    assert manifest["coverage"][0]["site"] == "ALMA"
    assert manifest["legacy_sources"][0]["path"].startswith("ALMA/04Apr/")
    root = zarr.open_group(store=zarr.storage.LocalStore(output), mode="r")
    assert root.attrs["dataset_id"] == "test-release-v0.1.0"


def test_cli_records_explicit_daily_repair_in_manifest(tmp_path):
    legacy_root = tmp_path / "weather_data_alltimes"
    source = legacy_root / "ALMA" / "04Apr"
    _write_legacy_partition(source, _with_invalid_duplicate_daily_record(_complete_partition()))

    tau_basis = tmp_path / "tau_basis"
    tb_basis = tmp_path / "tb_basis"
    basis = _basis()
    _write_basis(tau_basis, (basis.tau_mean, basis.tau_components))
    _write_basis(tb_basis, (basis.tb_mean, basis.tb_components))
    site_registry = tmp_path / "Telescope_Site_Matrix.csv"
    site_registry.write_text("Station\nALMA\n", encoding="utf-8")

    output = tmp_path / "repaired-release.zarr"
    assert main(
        [
            "--output",
            str(output),
            "--dataset-id",
            "test-repaired-release-v0.1.0",
            "--builder-revision",
            "test-revision",
            "--legacy-root",
            str(legacy_root),
            "--site-registry",
            str(site_registry),
            "--tau-basis",
            str(tau_basis),
            "--tb-basis",
            str(tb_basis),
            "--component-count",
            "2",
            "--repair-invalid-daily-records",
            "--all-partitions",
        ]
    ) == 0

    manifest = json.loads(
        (tmp_path / "repaired-release.zarr.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["coverage"][0]["legacy_invalid_daily_records_removed"] == 1
    assert manifest["validation"]["legacy_invalid_daily_records_removed"] == 1
