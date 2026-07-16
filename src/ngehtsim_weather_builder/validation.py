"""Validation helpers for legacy-to-Zarr weather dataset builds."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass

import numpy as np
import zarr

from .dataset import (
    NATIVE_SUMMARY_FIELDS,
    NATIVE_SUMMARY_FORMS,
    PcaBasis,
    _basis_from_root,
    native_summaries,
)
from .legacy import WeatherPartition, WeatherRecords, validate_partition


class DatasetValidationError(ValueError):
    """Raised when an imported dataset does not preserve its source records."""


@dataclass(frozen=True)
class PartitionCoverage:
    """Validated coverage details for one site-month partition."""

    first_date: str
    last_date: str
    years: tuple[int, ...]
    native_records: int
    daily_records: int


def _record_arrays(records: WeatherRecords) -> dict[str, np.ndarray]:
    arrays = {
        "year": records.year,
        "day": records.day,
        "tau_coefficients": records.tau_coefficients,
        "tb_coefficients": records.tb_coefficients,
        "pwv_mm": records.pwv_mm,
        "wind_speed_m_s": records.wind_speed_m_s,
        "surface_pressure_mbar": records.surface_pressure_mbar,
        "surface_temperature_k": records.surface_temperature_k,
    }
    if records.time_index is not None:
        arrays["time_index"] = records.time_index
    return arrays


def _first_and_last_date(records: WeatherRecords) -> tuple[str, str]:
    dates = np.column_stack((records.year, records.month, records.day))
    dates = np.unique(dates, axis=0)
    first = "{0:04d}-{1:02d}-{2:02d}".format(*dates[0])
    last = "{0:04d}-{1:02d}-{2:02d}".format(*dates[-1])
    return first, last


def validate_complete_calendar_coverage(partition: WeatherPartition) -> PartitionCoverage:
    """Require every calendar day for every covered year-month combination.

    Legacy files are organized by calendar month, so apparent gaps between one
    year's month and the next are expected. A missing day or an entire missing
    year within the covered range is not.
    """

    validate_partition(partition)
    records = partition.daily
    year_months = np.unique(np.column_stack((records.year, records.month)), axis=0)
    years = np.unique(records.year)
    expected_years = np.arange(years.min(), years.max() + 1, dtype=years.dtype)
    if not np.array_equal(years, expected_years):
        raise DatasetValidationError("Daily weather records have missing calendar years.")

    for year, month in year_months:
        observed_days = np.sort(records.day[(records.year == year) & (records.month == month)])
        expected_days = np.arange(1, monthrange(int(year), int(month))[1] + 1, dtype=observed_days.dtype)
        if not np.array_equal(observed_days, expected_days):
            raise DatasetValidationError(
                "Daily weather records are incomplete for {0:04d}-{1:02d}.".format(
                    int(year),
                    int(month),
                )
            )

    first_date, last_date = _first_and_last_date(records)
    return PartitionCoverage(
        first_date=first_date,
        last_date=last_date,
        years=tuple(int(year) for year in years),
        native_records=partition.native.count,
        daily_records=partition.daily.count,
    )


def validate_written_partition(
    root: zarr.Group,
    site: str,
    month: int,
    partition: WeatherPartition,
    basis: PcaBasis | None = None,
) -> None:
    """Verify that a Zarr partition is an exact representation of its source."""

    basis = _basis_from_root(root) if basis is None else basis
    validate_partition(partition, component_count=basis.component_count)
    path = "sites/{0}/months/{1:02d}".format(site, month)
    if path not in root:
        raise DatasetValidationError("Dataset is missing {0}.".format(path))

    month_group = root[path]
    if month_group.attrs.get("site") != site or month_group.attrs.get("month") != month:
        raise DatasetValidationError("Dataset partition metadata does not match its source.")

    for cadence, records in (("native", partition.native), ("daily", partition.daily)):
        group = month_group[cadence]
        expected_cadence = "three-hourly" if cadence == "native" else "daily"
        if group.attrs.get("cadence") != expected_cadence:
            raise DatasetValidationError("Dataset cadence metadata is invalid for {0}.".format(path))

        expected_arrays = _record_arrays(records)
        if set(group.array_keys()) != set(expected_arrays):
            raise DatasetValidationError("Dataset arrays do not match their source for {0}.".format(path))
        for name, expected in expected_arrays.items():
            stored = group[name]
            if np.dtype(stored.dtype) != np.dtype(expected.dtype):
                raise DatasetValidationError(
                    "Dataset dtype for {0}/{1} does not match its source.".format(path, name)
                )
            try:
                np.testing.assert_array_equal(stored[:], expected)
            except AssertionError as error:
                raise DatasetValidationError(
                    "Dataset values for {0}/{1} do not match their source.".format(path, name)
                ) from error

    if "native_summary" not in month_group:
        raise DatasetValidationError("Dataset is missing native summaries for {0}.".format(path))
    summary_group = month_group["native_summary"]
    if set(summary_group.group_keys()) != set(NATIVE_SUMMARY_FORMS):
        raise DatasetValidationError("Dataset native summary forms are invalid for {0}.".format(path))

    expected_summaries = native_summaries(partition, basis)
    for form, expected_summary in expected_summaries.items():
        form_group = summary_group[form]
        if set(form_group.array_keys()) != set(NATIVE_SUMMARY_FIELDS):
            raise DatasetValidationError(
                "Dataset native summary arrays are invalid for {0}/{1}.".format(path, form)
            )
        for name, expected in expected_summary.items():
            stored = form_group[name]
            if np.dtype(stored.dtype) != np.dtype(np.float64):
                raise DatasetValidationError(
                    "Dataset native summary dtype for {0}/{1}/{2} is invalid.".format(
                        path,
                        form,
                        name,
                    )
                )
            try:
                np.testing.assert_array_equal(stored[:], expected)
            except AssertionError as error:
                raise DatasetValidationError(
                    "Dataset native summary values for {0}/{1}/{2} do not match their "
                    "source.".format(path, form, name)
                ) from error
