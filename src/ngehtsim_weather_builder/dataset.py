"""Canonical Zarr dataset creation for ngehtsim weather products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import zarr

from .legacy import WeatherPartition, WeatherRecords, validate_partition


SCHEMA_VERSION = "0.2.0"
NATIVE_SAMPLES_PER_DAY = 8
NATIVE_SUMMARY_FORMS = ("mean", "median", "good", "bad")
NATIVE_SUMMARY_FIELDS = (
    "opacity",
    "brightness_temperature",
    "pwv_mm",
    "wind_speed_m_s",
    "surface_pressure_mbar",
    "surface_temperature_k",
)


@dataclass(frozen=True)
class PcaBasis:
    """The PCA bases needed to interpret stored weather coefficients."""

    tau_mean: np.ndarray
    tau_components: np.ndarray
    tb_mean: np.ndarray
    tb_components: np.ndarray

    @property
    def component_count(self) -> int:
        return int(self.tau_components.shape[0])

    @property
    def spectral_length(self) -> int:
        return int(self.tau_mean.size)


def _load_basis(directory: Path, component_count: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.loadtxt(directory / "spectrum_mean.txt", unpack=True)
    components = np.stack(
        [
            np.loadtxt(directory / "spectrum_{0:04d}.txt".format(index), unpack=True)
            for index in range(component_count)
        ]
    )
    return mean, components


def load_pca_basis(
    tau_directory: str | Path,
    tb_directory: str | Path,
    component_count: int = 40,
) -> PcaBasis:
    """Load a PCA basis from legacy eigenspectra directories."""

    tau_mean, tau_components = _load_basis(Path(tau_directory), component_count)
    tb_mean, tb_components = _load_basis(Path(tb_directory), component_count)
    basis = PcaBasis(
        tau_mean=np.asarray(tau_mean, dtype=np.float64),
        tau_components=np.asarray(tau_components, dtype=np.float64),
        tb_mean=np.asarray(tb_mean, dtype=np.float64),
        tb_components=np.asarray(tb_components, dtype=np.float64),
    )
    _validate_basis(basis)
    return basis


def _validate_basis(basis: PcaBasis) -> None:
    if basis.tau_mean.ndim != 1 or basis.tb_mean.ndim != 1:
        raise ValueError("PCA mean spectra must be one-dimensional.")
    if basis.tau_components.ndim != 2 or basis.tb_components.ndim != 2:
        raise ValueError("PCA component arrays must be two-dimensional.")
    if basis.tau_mean.shape != basis.tb_mean.shape:
        raise ValueError("Tau and Tb mean spectra must have the same length.")
    expected_shape = (basis.component_count, basis.spectral_length)
    if basis.tau_components.shape != expected_shape or basis.tb_components.shape != expected_shape:
        raise ValueError("PCA component shapes do not match the mean spectra.")


def _chunks(data: np.ndarray) -> tuple[int, ...]:
    if data.ndim == 1:
        return (data.shape[0],)
    return (data.shape[0], data.shape[1])


def initialize_dataset(
    output_path: str | Path,
    basis: PcaBasis,
    frequency_ghz: np.ndarray,
    metadata: Mapping[str, str] | None = None,
) -> zarr.Group:
    """Create a new, empty weather dataset with PCA metadata."""

    _validate_basis(basis)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing dataset: {0}".format(output))

    frequency = np.asarray(frequency_ghz, dtype=np.float64)
    if frequency.ndim != 1 or frequency.shape != (basis.spectral_length,):
        raise ValueError("The frequency grid must match the PCA spectral length.")

    root = zarr.open_group(
        store=zarr.storage.LocalStore(output),
        mode="w",
        zarr_format=3,
    )
    root.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "native_time_step_hours": 3,
            "native_samples_per_day": NATIVE_SAMPLES_PER_DAY,
            "native_summary_forms": list(NATIVE_SUMMARY_FORMS),
            "native_summary_derivation": (
                "Reconstruct native spectra and reduce physical quantities by "
                "three-hour UTC time index across the full site-month history."
            ),
            "daily_derivation": (
                "Reconstruct native spectra, average physical tau and Tb spectra, "
                "then project the daily averages onto the PCA basis."
            ),
        }
    )
    if metadata:
        root.attrs.update(dict(metadata))

    root.create_array("frequency_ghz", data=frequency, chunks=_chunks(frequency))
    pca = root.require_group("pca")
    for quantity, mean, components in (
        ("tau", basis.tau_mean, basis.tau_components),
        ("tb", basis.tb_mean, basis.tb_components),
    ):
        group = pca.require_group(quantity)
        group.create_array("mean", data=mean, chunks=_chunks(mean))
        group.create_array("components", data=components, chunks=(1, components.shape[1]))
    return root


def _write_records(group: zarr.Group, records: WeatherRecords) -> None:
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

    for name, values in arrays.items():
        data = np.asarray(values)
        group.create_array(name, data=data, chunks=_chunks(data))


def _native_reducer(form: str):
    if form == "mean":
        return np.nanmean
    if form == "median":
        return np.nanmedian
    if form == "good":
        return lambda values, axis: np.nanpercentile(values, 15.87, axis=axis)
    if form == "bad":
        return lambda values, axis: np.nanpercentile(values, 84.13, axis=axis)
    raise ValueError("Unsupported native weather summary form: {0}".format(form))


def native_summaries(
    partition: WeatherPartition,
    basis: PcaBasis,
) -> dict[str, dict[str, np.ndarray]]:
    """Reduce native records into physical three-hour site-month summaries."""

    _validate_basis(basis)
    validate_partition(partition, component_count=basis.component_count)
    native = partition.native
    if native.time_index is None:
        raise ValueError("Native weather records require a time index.")

    values = {
        "opacity": np.power(
            10.0,
            basis.tau_mean + native.tau_coefficients @ basis.tau_components,
        ),
        "brightness_temperature": (
            basis.tb_mean + native.tb_coefficients @ basis.tb_components
        ),
        "pwv_mm": native.pwv_mm,
        "wind_speed_m_s": native.wind_speed_m_s,
        "surface_pressure_mbar": native.surface_pressure_mbar,
        "surface_temperature_k": native.surface_temperature_k,
    }
    summaries: dict[str, dict[str, np.ndarray]] = {}
    for form in NATIVE_SUMMARY_FORMS:
        reducer = _native_reducer(form)
        summary = {}
        for name, source in values.items():
            reduced = []
            for time_index in range(NATIVE_SAMPLES_PER_DAY):
                mask = native.time_index == time_index
                if not np.any(mask):
                    raise ValueError(
                        "Native weather records are missing time index {0}.".format(time_index)
                    )
                reduced.append(reducer(source[mask], axis=0))
            summary[name] = np.asarray(reduced, dtype=np.float64)
        summaries[form] = summary
    return summaries


def _basis_from_root(root: zarr.Group) -> PcaBasis:
    """Load the stored PCA basis for direct API calls without an explicit basis."""

    return PcaBasis(
        tau_mean=np.asarray(root["pca/tau/mean"][:], dtype=np.float64),
        tau_components=np.asarray(root["pca/tau/components"][:], dtype=np.float64),
        tb_mean=np.asarray(root["pca/tb/mean"][:], dtype=np.float64),
        tb_components=np.asarray(root["pca/tb/components"][:], dtype=np.float64),
    )


def _write_native_summaries(
    group: zarr.Group,
    partition: WeatherPartition,
    basis: PcaBasis,
) -> None:
    for form, summary in native_summaries(partition, basis).items():
        form_group = group.require_group(form)
        for name, values in summary.items():
            form_group.create_array(name, data=values, chunks=_chunks(values))


def write_partition(
    root: zarr.Group,
    site: str,
    month: int,
    partition: WeatherPartition,
    basis: PcaBasis | None = None,
) -> None:
    """Write one validated site-month partition into an initialized dataset."""

    if not site:
        raise ValueError("A site name is required.")
    if month < 1 or month > 12:
        raise ValueError("Month must be an integer from 1 through 12.")

    basis = _basis_from_root(root) if basis is None else basis
    _validate_basis(basis)
    validate_partition(partition, component_count=basis.component_count)
    for records in (partition.native, partition.daily):
        if not np.all(records.month == month):
            raise ValueError("Partition record months do not match the requested month.")

    path = "sites/{0}/months/{1:02d}".format(site, month)
    if path in root:
        raise ValueError("Dataset already contains {0}.".format(path))

    month_group = root.require_group(path)
    month_group.attrs.update({"site": site, "month": month})
    native_group = month_group.require_group("native")
    native_group.attrs.update({"cadence": "three-hourly"})
    _write_records(native_group, partition.native)
    daily_group = month_group.require_group("daily")
    daily_group.attrs.update({"cadence": "daily"})
    _write_records(daily_group, partition.daily)
    _write_native_summaries(month_group.require_group("native_summary"), partition, basis)
