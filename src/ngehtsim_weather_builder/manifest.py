"""Immutable, machine-readable provenance manifests for weather datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset import NATIVE_SUMMARY_FORMS, PcaBasis, SCHEMA_VERSION
from .validation import PartitionCoverage


MANIFEST_VERSION = "1"
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ImportedPartition:
    """A partition successfully copied from a legacy directory."""

    site: str
    month: int
    source: Path
    coverage: PartitionCoverage
    removed_daily_records: int = 0


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Return a checksum that includes an array's shape, dtype, and values."""

    data = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(repr(data.shape).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def _directory_fingerprint(directory: Path) -> dict[str, object]:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return {
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def _source_files(legacy_root: Path, partitions: Iterable[ImportedPartition]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for imported in sorted(partitions, key=lambda item: (item.site, item.month)):
        try:
            relative_source = imported.source.relative_to(legacy_root)
        except ValueError as error:
            raise ValueError(
                "Partition source {0} is outside legacy root {1}.".format(
                    imported.source,
                    legacy_root,
                )
            ) from error
        for path in sorted(imported.source.glob("*.txt")):
            sources.append(
                {
                    "path": str(relative_source / path.name),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sources


def build_manifest(
    *,
    dataset_id: str,
    builder_revision: str,
    legacy_root: str | Path,
    site_registry: str | Path,
    tau_basis_directory: str | Path,
    tb_basis_directory: str | Path,
    basis: PcaBasis,
    frequency_ghz: np.ndarray,
    partitions: Iterable[ImportedPartition],
) -> dict[str, object]:
    """Build a serializable manifest after all dataset checks have passed."""

    legacy_root = Path(legacy_root).resolve()
    site_registry = Path(site_registry).resolve()
    tau_basis_directory = Path(tau_basis_directory).resolve()
    tb_basis_directory = Path(tb_basis_directory).resolve()
    imported = tuple(partitions)
    removed_total = sum(item.removed_daily_records for item in imported)
    checks = [
        "legacy binary-record validation",
        "native three-hour timestamp completeness",
        "native and daily date agreement",
        "complete calendar-month coverage",
        "exact Zarr array and dtype comparison",
        "native physical-summary array comparison",
    ]
    if removed_total:
        checks.append("validated removal of malformed legacy daily records")

    return {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": {
            "id": dataset_id,
            "schema_version": SCHEMA_VERSION,
            "native_time_step_hours": 3,
            "native_samples_per_day": 8,
            "native_summary_forms": list(NATIVE_SUMMARY_FORMS),
            "frequency_ghz": {
                "count": int(frequency_ghz.size),
                "sha256": sha256_array(frequency_ghz),
            },
            "pca": {
                "component_count": basis.component_count,
                "spectral_length": basis.spectral_length,
                "tau_mean_sha256": sha256_array(basis.tau_mean),
                "tau_components_sha256": sha256_array(basis.tau_components),
                "tb_mean_sha256": sha256_array(basis.tb_mean),
                "tb_components_sha256": sha256_array(basis.tb_components),
            },
        },
        "builder": {"revision": builder_revision},
        "site_registry": {
            "filename": site_registry.name,
            "sha256": sha256_file(site_registry),
        },
        "pca_basis_sources": {
            "tau": _directory_fingerprint(tau_basis_directory),
            "tb": _directory_fingerprint(tb_basis_directory),
        },
        "legacy_sources": _source_files(legacy_root, imported),
        "coverage": [
            {
                "site": item.site,
                "month": item.month,
                "first_date": item.coverage.first_date,
                "last_date": item.coverage.last_date,
                "years": list(item.coverage.years),
                "native_records": item.coverage.native_records,
                "daily_records": item.coverage.daily_records,
                "legacy_invalid_daily_records_removed": item.removed_daily_records,
                "calendar_coverage": "complete",
            }
            for item in sorted(imported, key=lambda item: (item.site, item.month))
        ],
        "validation": {
            "status": "passed",
            "legacy_invalid_daily_records_removed": removed_total,
            "checks": checks,
        },
    }


def write_manifest(path: str | Path, manifest: dict[str, object]) -> None:
    """Write a readable JSON manifest without replacing an existing file."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing manifest: {0}".format(destination))
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
