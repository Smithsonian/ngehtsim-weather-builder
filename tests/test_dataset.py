import numpy as np
import pytest
import zarr

from ngehtsim_weather_builder.dataset import PcaBasis, initialize_dataset, write_partition
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
    assert reopened.attrs["schema_version"] == "0.1.0"
    assert reopened.attrs["dataset_id"] == "test-2026.1"
    np.testing.assert_array_equal(reopened["frequency_ghz"][:], np.arange(4))
    np.testing.assert_array_equal(
        reopened["sites/ALMA/months/04/native/time_index"][:],
        np.arange(8),
    )
    assert "time_index" not in reopened["sites/ALMA/months/04/daily"]
    assert reopened["sites/ALMA/months/04/native/tau_coefficients"].dtype == np.dtype("<f2")


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
