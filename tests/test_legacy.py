import struct

import numpy as np
import pytest

from ngehtsim_weather_builder.legacy import (
    LegacyFormatError,
    read_legacy_partition,
    repair_invalid_daily_records,
)


def _dtype(component_count, with_time, coefficients):
    fields = [("year", "<i2"), ("month", "i1"), ("day", "i1")]
    if with_time:
        fields.append(("time_index", "i1"))
    if coefficients:
        fields.append(("value", "<f2", (component_count,)))
    else:
        fields.append(("value", "<f8"))
    return np.dtype(fields)


def _write_binary(path, records, component_count, with_time, coefficients):
    dtype = _dtype(component_count, with_time, coefficients)
    assert records.dtype == dtype
    path.write_bytes(struct.pack("<H", dtype.itemsize) + records.tobytes())


def _coordinates(count, with_time):
    values = {
        "year": np.full(count, 2017, dtype="<i2"),
        "month": np.full(count, 4, dtype="i1"),
        "day": np.full(count, 11, dtype="i1"),
    }
    if with_time:
        values["time_index"] = np.arange(count, dtype="i1")
    return values


def _write_partition(directory, component_count=2, native_count=8, daily_count=1):
    directory.mkdir()

    for suffix, with_time, count in (("_alltimes", True, native_count), ("", False, daily_count)):
        coordinates = _coordinates(count, with_time)
        for name, offset in (("tau", 0.0), ("Tb", 10.0)):
            records = np.zeros(count, dtype=_dtype(component_count, with_time, True))
            for key, values in coordinates.items():
                records[key] = values
            records["value"] = (
                np.arange(count * component_count, dtype=np.float16).reshape(count, component_count)
                + offset
            )
            _write_binary(
                directory / "{0}{1}.txt".format(name, suffix),
                records,
                component_count,
                with_time,
                coefficients=True,
            )

        for index, name in enumerate(("PWV", "windspeed", "Pbase", "Tbase")):
            records = np.zeros(count, dtype=_dtype(component_count, with_time, False))
            for key, values in coordinates.items():
                records[key] = values
            records["value"] = np.arange(count, dtype=np.float64) + index
            _write_binary(
                directory / "{0}{1}.txt".format(name, suffix),
                records,
                component_count,
                with_time,
                coefficients=False,
            )


def test_reads_valid_legacy_partition(tmp_path):
    source = tmp_path / "04Apr"
    _write_partition(source)

    partition = read_legacy_partition(source, component_count=2)

    assert partition.native.count == 8
    assert partition.daily.count == 1
    np.testing.assert_array_equal(partition.native.time_index, np.arange(8))
    np.testing.assert_allclose(partition.native.tb_coefficients[0], [10.0, 11.0])


def test_rejects_incomplete_native_day(tmp_path):
    source = tmp_path / "04Apr"
    _write_partition(source, native_count=7)

    with pytest.raises(LegacyFormatError, match="exactly 8 samples"):
        read_legacy_partition(source, component_count=2)


def test_rejects_mismatched_coordinate_rows(tmp_path):
    source = tmp_path / "04Apr"
    _write_partition(source)

    target = source / "windspeed_alltimes.txt"
    dtype = _dtype(component_count=2, with_time=True, coefficients=False)
    records = np.frombuffer(target.read_bytes()[2:], dtype=dtype).copy()
    records["day"][0] = 12
    target.write_bytes(struct.pack("<H", dtype.itemsize) + records.tobytes())

    with pytest.raises(LegacyFormatError, match="do not match"):
        read_legacy_partition(source, component_count=2)


def test_repairs_only_redundant_invalid_daily_records(tmp_path):
    source = tmp_path / "04Apr"
    _write_partition(source, daily_count=2)

    target = source / "tau.txt"
    dtype = _dtype(component_count=2, with_time=False, coefficients=True)
    records = np.frombuffer(target.read_bytes()[2:], dtype=dtype).copy()
    records["value"][1] = np.nan
    target.write_bytes(struct.pack("<H", dtype.itemsize) + records.tobytes())

    repaired = repair_invalid_daily_records(source, component_count=2)

    assert repaired.removed_daily_records == 1
    assert repaired.partition.daily.count == 1
    np.testing.assert_array_equal(repaired.partition.daily.day, [11])


def test_repair_rejects_missing_daily_coverage(tmp_path):
    source = tmp_path / "04Apr"
    _write_partition(source)

    target = source / "tau.txt"
    dtype = _dtype(component_count=2, with_time=False, coefficients=True)
    records = np.frombuffer(target.read_bytes()[2:], dtype=dtype).copy()
    records["value"][:] = np.nan
    target.write_bytes(struct.pack("<H", dtype.itemsize) + records.tobytes())

    with pytest.raises(LegacyFormatError, match="cannot be empty"):
        repair_invalid_daily_records(source, component_count=2)
