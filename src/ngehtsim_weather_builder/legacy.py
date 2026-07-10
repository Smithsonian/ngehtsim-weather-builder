"""Readers and validation for the legacy ngehtsim weather binary format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


NATIVE_SAMPLES_PER_DAY = 8


class LegacyFormatError(ValueError):
    """Raised when a legacy weather partition is malformed or incomplete."""


@dataclass(frozen=True)
class WeatherRecords:
    """One cadence of weather records for a site and calendar month."""

    year: np.ndarray
    month: np.ndarray
    day: np.ndarray
    tau_coefficients: np.ndarray
    tb_coefficients: np.ndarray
    pwv_mm: np.ndarray
    wind_speed_m_s: np.ndarray
    surface_pressure_mbar: np.ndarray
    surface_temperature_k: np.ndarray
    time_index: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(self.year.size)


@dataclass(frozen=True)
class WeatherPartition:
    """Native and daily records for one legacy site-month directory."""

    native: WeatherRecords
    daily: WeatherRecords


def _record_dtype(component_count: int, with_time: bool, coefficients: bool) -> np.dtype:
    fields: list[tuple[object, ...]] = [
        ("year", "<i2"),
        ("month", "i1"),
        ("day", "i1"),
    ]
    if with_time:
        fields.append(("time_index", "i1"))
    if coefficients:
        fields.append(("value", "<f2", (component_count,)))
    else:
        fields.append(("value", "<f8"))
    return np.dtype(fields)


def _read_binary(path: Path, dtype: np.dtype) -> np.ndarray:
    contents = path.read_bytes()
    if len(contents) < 2:
        raise LegacyFormatError("{0} does not contain a record header.".format(path))

    record_size = struct.unpack_from("<H", contents)[0]
    if record_size != dtype.itemsize:
        raise LegacyFormatError(
            "{0} declares {1}-byte records; expected {2} bytes.".format(
                path,
                record_size,
                dtype.itemsize,
            )
        )

    payload = contents[2:]
    if len(payload) % record_size != 0:
        raise LegacyFormatError(
            "{0} has a truncated record payload ({1} bytes).".format(
                path,
                len(payload),
            )
        )

    return np.frombuffer(payload, dtype=dtype)


def _coordinates_match(reference: np.ndarray, candidate: np.ndarray, with_time: bool) -> bool:
    names = ["year", "month", "day"]
    if with_time:
        names.append("time_index")
    return all(np.array_equal(reference[name], candidate[name]) for name in names)


def _read_records(directory: Path, suffix: str, with_time: bool, component_count: int) -> WeatherRecords:
    atmospheric_dtype = _record_dtype(component_count, with_time, coefficients=True)
    scalar_dtype = _record_dtype(component_count, with_time, coefficients=False)

    tau = _read_binary(directory / "tau{0}.txt".format(suffix), atmospheric_dtype)
    tb = _read_binary(directory / "Tb{0}.txt".format(suffix), atmospheric_dtype)
    scalar_files = {
        "pwv_mm": "PWV{0}.txt".format(suffix),
        "wind_speed_m_s": "windspeed{0}.txt".format(suffix),
        "surface_pressure_mbar": "Pbase{0}.txt".format(suffix),
        "surface_temperature_k": "Tbase{0}.txt".format(suffix),
    }
    scalar_data = {
        name: _read_binary(directory / filename, scalar_dtype)
        for name, filename in scalar_files.items()
    }

    for name, values in [("Tb", tb), *scalar_data.items()]:
        if not _coordinates_match(tau, values, with_time):
            raise LegacyFormatError(
                "Coordinate rows in {0} do not match tau{1}.txt.".format(
                    name,
                    suffix,
                )
            )

    time_index = tau["time_index"] if with_time else None
    return WeatherRecords(
        year=tau["year"],
        month=tau["month"],
        day=tau["day"],
        time_index=time_index,
        tau_coefficients=tau["value"],
        tb_coefficients=tb["value"],
        pwv_mm=scalar_data["pwv_mm"]["value"],
        wind_speed_m_s=scalar_data["wind_speed_m_s"]["value"],
        surface_pressure_mbar=scalar_data["surface_pressure_mbar"]["value"],
        surface_temperature_k=scalar_data["surface_temperature_k"]["value"],
    )


def _validate_record_arrays(records: WeatherRecords, component_count: int) -> None:
    expected_shapes = {
        "month": (records.count,),
        "day": (records.count,),
        "tau_coefficients": (records.count, component_count),
        "tb_coefficients": (records.count, component_count),
        "pwv_mm": (records.count,),
        "wind_speed_m_s": (records.count,),
        "surface_pressure_mbar": (records.count,),
        "surface_temperature_k": (records.count,),
    }
    for name, shape in expected_shapes.items():
        if getattr(records, name).shape != shape:
            raise LegacyFormatError(
                "{0} has shape {1}; expected {2}.".format(
                    name,
                    getattr(records, name).shape,
                    shape,
                )
            )

    if records.count == 0:
        raise LegacyFormatError("A weather record stream cannot be empty.")
    if np.any(records.month < 1) or np.any(records.month > 12):
        raise LegacyFormatError("Weather records contain an invalid month.")
    if np.any(records.day < 1) or np.any(records.day > 31):
        raise LegacyFormatError("Weather records contain an invalid day.")

    scalar_fields = (
        records.pwv_mm,
        records.wind_speed_m_s,
        records.surface_pressure_mbar,
        records.surface_temperature_k,
    )
    if not all(np.all(np.isfinite(values)) for values in scalar_fields):
        raise LegacyFormatError("Weather records contain non-finite scalar values.")
    coefficient_fields = (records.tau_coefficients, records.tb_coefficients)
    if not all(np.all(np.isfinite(values)) for values in coefficient_fields):
        raise LegacyFormatError("Weather records contain non-finite PCA coefficients.")


def _date_rows(records: WeatherRecords) -> np.ndarray:
    return np.column_stack((records.year, records.month, records.day))


def _validate_partition(partition: WeatherPartition, component_count: int) -> None:
    native = partition.native
    daily = partition.daily
    _validate_record_arrays(native, component_count)
    _validate_record_arrays(daily, component_count)

    if native.time_index is None:
        raise LegacyFormatError("Native weather records require a time index.")
    if daily.time_index is not None:
        raise LegacyFormatError("Daily weather records cannot include a time index.")
    if np.any(native.time_index < 0) or np.any(native.time_index >= NATIVE_SAMPLES_PER_DAY):
        raise LegacyFormatError("Native weather records contain an invalid time index.")

    native_dates = _date_rows(native)
    unique_native_dates, native_counts = np.unique(native_dates, axis=0, return_counts=True)
    if np.any(native_counts != NATIVE_SAMPLES_PER_DAY):
        raise LegacyFormatError(
            "Every native date must contain exactly {0} samples.".format(
                NATIVE_SAMPLES_PER_DAY
            )
        )

    for date in unique_native_dates:
        indices = np.all(native_dates == date, axis=1)
        times = np.sort(native.time_index[indices])
        expected = np.arange(NATIVE_SAMPLES_PER_DAY, dtype=times.dtype)
        if not np.array_equal(times, expected):
            raise LegacyFormatError(
                "Native time indices must be the complete range 0 through {0}.".format(
                    NATIVE_SAMPLES_PER_DAY - 1
                )
            )

    daily_dates = _date_rows(daily)
    unique_daily_dates, daily_counts = np.unique(daily_dates, axis=0, return_counts=True)
    if np.any(daily_counts != 1):
        raise LegacyFormatError("Daily weather records contain duplicate dates.")
    if not np.array_equal(unique_native_dates, unique_daily_dates):
        raise LegacyFormatError("Native and daily weather records cover different dates.")


def read_legacy_partition(directory: str | Path, component_count: int = 40) -> WeatherPartition:
    """Read and validate one legacy site-month directory."""

    source = Path(directory)
    partition = WeatherPartition(
        native=_read_records(source, "_alltimes", with_time=True, component_count=component_count),
        daily=_read_records(source, "", with_time=False, component_count=component_count),
    )
    _validate_partition(partition, component_count)
    return partition


def validate_partition(partition: WeatherPartition, component_count: int | None = None) -> None:
    """Validate native and daily record coverage before a dataset write."""

    count = component_count
    if count is None:
        count = int(partition.native.tau_coefficients.shape[1])
    _validate_partition(partition, count)
