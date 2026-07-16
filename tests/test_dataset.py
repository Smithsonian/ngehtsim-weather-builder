import numpy as np
import pytest
import zarr

from ngehtsim_weather_builder.dataset import (
    NATIVE_SUMMARY_FIELDS,
    NATIVE_SUMMARY_FORMS,
    PcaBasis,
    initialize_dataset,
    native_summaries,
    write_partition,
)
from ngehtsim_weather_builder.legacy import WeatherPartition, WeatherRecords


def _records(with_time):
    count = 8 if with_time else 1
    return WeatherRecords(
        year=np.full(count, 2017, dtype="<i2"),
        month=np.full(count, 4, dtype="i1"),
        day=np.full(count, 11, dtype="i1"),
        time_index=np.arange(count, dtype="i1") if with_time else None,
        tau_coefficients=np.arange(count * 2, dtype=np.float16).reshape(count, 2),
        tb_coefficients=np.arange(count * 2, dtype=np.float16).reshape(count, 2) + 10,
        pwv_mm=np.arange(count, dtype=np.float64),
        wind_speed_m_s=np.arange(count, dtype=np.float64) + 1,
        surface_pressure_mbar=np.arange(count, dtype=np.float64) + 2,
        surface_temperature_k=np.arange(count, dtype=np.float64) + 3,
    )


def _basis():
    return PcaBasis(
        tau_mean=np.array([1.0, 2.0, 3.0, 4.0]),
        tau_components=np.arange(8, dtype=np.float64).reshape(2, 4),
        tb_mean=np.array([5.0, 6.0, 7.0, 8.0]),
        tb_components=np.arange(8, 16, dtype=np.float64).reshape(2, 4),
    )


def _two_day_partition():
    native = _records(True)
    second_native = _records(True)
    second_native = WeatherRecords(
        year=second_native.year,
        month=second_native.month,
        day=second_native.day + 1,
        time_index=second_native.time_index,
        tau_coefficients=second_native.tau_coefficients + 1,
        tb_coefficients=second_native.tb_coefficients + 1,
        pwv_mm=second_native.pwv_mm + 10,
        wind_speed_m_s=second_native.wind_speed_m_s + 10,
        surface_pressure_mbar=second_native.surface_pressure_mbar + 10,
        surface_temperature_k=second_native.surface_temperature_k + 10,
    )
    daily = _records(False)
    second_daily = WeatherRecords(
        year=daily.year,
        month=daily.month,
        day=daily.day + 1,
        time_index=None,
        tau_coefficients=daily.tau_coefficients + 1,
        tb_coefficients=daily.tb_coefficients + 1,
        pwv_mm=daily.pwv_mm + 10,
        wind_speed_m_s=daily.wind_speed_m_s + 10,
        surface_pressure_mbar=daily.surface_pressure_mbar + 10,
        surface_temperature_k=daily.surface_temperature_k + 10,
    )
    return WeatherPartition(
        native=WeatherRecords(
            year=np.concatenate((native.year, second_native.year)),
            month=np.concatenate((native.month, second_native.month)),
            day=np.concatenate((native.day, second_native.day)),
            time_index=np.concatenate((native.time_index, second_native.time_index)),
            tau_coefficients=np.concatenate(
                (native.tau_coefficients, second_native.tau_coefficients)
            ),
            tb_coefficients=np.concatenate((native.tb_coefficients, second_native.tb_coefficients)),
            pwv_mm=np.concatenate((native.pwv_mm, second_native.pwv_mm)),
            wind_speed_m_s=np.concatenate((native.wind_speed_m_s, second_native.wind_speed_m_s)),
            surface_pressure_mbar=np.concatenate(
                (native.surface_pressure_mbar, second_native.surface_pressure_mbar)
            ),
            surface_temperature_k=np.concatenate(
                (native.surface_temperature_k, second_native.surface_temperature_k)
            ),
        ),
        daily=WeatherRecords(
            year=np.concatenate((daily.year, second_daily.year)),
            month=np.concatenate((daily.month, second_daily.month)),
            day=np.concatenate((daily.day, second_daily.day)),
            time_index=None,
            tau_coefficients=np.concatenate((daily.tau_coefficients, second_daily.tau_coefficients)),
            tb_coefficients=np.concatenate((daily.tb_coefficients, second_daily.tb_coefficients)),
            pwv_mm=np.concatenate((daily.pwv_mm, second_daily.pwv_mm)),
            wind_speed_m_s=np.concatenate((daily.wind_speed_m_s, second_daily.wind_speed_m_s)),
            surface_pressure_mbar=np.concatenate(
                (daily.surface_pressure_mbar, second_daily.surface_pressure_mbar)
            ),
            surface_temperature_k=np.concatenate(
                (daily.surface_temperature_k, second_daily.surface_temperature_k)
            ),
        ),
    )


def test_writes_site_month_partition(tmp_path):
    output = tmp_path / "weather.zarr"
    root = initialize_dataset(
        output,
        _basis(),
        np.arange(4, dtype=np.float64),
        metadata={"dataset_id": "test-2026.1"},
    )
    partition = WeatherPartition(native=_records(True), daily=_records(False))

    write_partition(root, "ALMA", 4, partition)

    reopened = zarr.open_group(store=zarr.storage.LocalStore(output), mode="r")
    assert reopened.attrs["schema_version"] == "0.2.0"
    assert reopened.attrs["dataset_id"] == "test-2026.1"
    np.testing.assert_array_equal(reopened["frequency_ghz"][:], np.arange(4))
    np.testing.assert_array_equal(
        reopened["sites/ALMA/months/04/native/time_index"][:],
        np.arange(8),
    )
    assert "time_index" not in reopened["sites/ALMA/months/04/daily"]
    assert reopened["sites/ALMA/months/04/native/tau_coefficients"].dtype == np.dtype("<f2")


def test_writes_physical_native_summary_products(tmp_path):
    output = tmp_path / "weather.zarr"
    basis = _basis()
    root = initialize_dataset(output, basis, np.arange(4, dtype=np.float64))
    partition = _two_day_partition()

    write_partition(root, "ALMA", 4, partition, basis=basis)

    expected = native_summaries(partition, basis)
    stored = root["sites/ALMA/months/04/native_summary"]
    assert set(stored.group_keys()) == set(NATIVE_SUMMARY_FORMS)
    for form, values in expected.items():
        assert set(stored[form].array_keys()) == set(NATIVE_SUMMARY_FIELDS)
        for name, expected_values in values.items():
            assert stored[form][name].dtype == np.dtype(np.float64)
            np.testing.assert_array_equal(stored[form][name][:], expected_values)

    native = partition.native
    native_opacity = np.power(
        10.0,
        basis.tau_mean + native.tau_coefficients @ basis.tau_components,
    )
    expected_good = np.nanpercentile(native_opacity[native.time_index == 0], 15.87, axis=0)
    np.testing.assert_array_equal(stored["good"]["opacity"][0], expected_good)


def test_refuses_to_overwrite_a_dataset(tmp_path):
    output = tmp_path / "weather.zarr"
    initialize_dataset(output, _basis(), np.arange(4, dtype=np.float64))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        initialize_dataset(output, _basis(), np.arange(4, dtype=np.float64))


def test_refuses_incomplete_native_records(tmp_path):
    output = tmp_path / "weather.zarr"
    root = initialize_dataset(output, _basis(), np.arange(4, dtype=np.float64))
    incomplete_native = _records(True)
    incomplete_native = WeatherRecords(
        **{
            name: getattr(incomplete_native, name)[:7]
            if isinstance(getattr(incomplete_native, name), np.ndarray)
            else getattr(incomplete_native, name)
            for name in incomplete_native.__dataclass_fields__
        }
    )
    partition = WeatherPartition(native=incomplete_native, daily=_records(False))

    with pytest.raises(ValueError, match="exactly 8 samples"):
        write_partition(root, "ALMA", 4, partition)
