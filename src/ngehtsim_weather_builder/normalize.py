"""Create clean, cutoff-bounded legacy weather archives for release building.

The legacy weather format stores daily and native three-hour records together
under each ``SITE/MMMmm`` directory. Historic archives can contain incomplete
future months or redundant malformed daily records. This module creates a new
archive from the native records without modifying the source directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

from .legacy import (
    NormalizedLegacyPartition,
    normalize_legacy_partition,
    read_legacy_partition,
    write_legacy_partition,
)
from .validation import PartitionCoverage, validate_complete_calendar_coverage


_MONTH_DIRECTORY = re.compile(
    r"^(?P<month>0[1-9]|1[0-2])(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$"
)
_DAILY_FILENAMES = ("tau.txt", "Tb.txt", "PWV.txt", "windspeed.txt", "Pbase.txt", "Tbase.txt")


@dataclass(frozen=True)
class NormalizedPartition:
    """One site-month output written by the normalization command."""

    site: str
    month: int
    source: Path
    coverage: PartitionCoverage
    removed_native_records: int
    removed_daily_records: int


def _parse_cutoff(value: str) -> date:
    """Parse a required ISO-8601 month-end cutoff date."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Cutoff dates must use ISO-8601 form YYYY-MM-DD."
        ) from error
    if parsed.day != _month_end(parsed.year, parsed.month):
        raise argparse.ArgumentTypeError(
            "The release cutoff must be the final day of its calendar month."
        )
    return parsed


def _month_end(year: int, month: int) -> int:
    """Return the number of days in a calendar month without external dependencies."""

    if month == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def _discover_partitions(root: Path) -> list[tuple[str, int, Path]]:
    """Discover every recognized ``SITE/MMMmm`` directory below ``root``."""

    partitions: list[tuple[str, int, Path]] = []
    for site_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for month_directory in sorted(path for path in site_directory.iterdir() if path.is_dir()):
            match = _MONTH_DIRECTORY.fullmatch(month_directory.name)
            if match is not None:
                partitions.append(
                    (site_directory.name, int(match.group("month")), month_directory)
                )
    if not partitions:
        raise ValueError("No legacy SITE/MMMmm directories were found below {0}.".format(root))
    return partitions


def _registry_sites(path: Path) -> set[str]:
    """Read the canonical site names from a UTF-8 CSV site registry."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "Name" not in reader.fieldnames:
            raise ValueError("Site registry must contain a Name column: {0}".format(path))
        sites = {row["Name"] for row in reader if row.get("Name")}
    if not sites:
        raise ValueError("Site registry does not contain any site names: {0}".format(path))
    return sites


def _validate_partition_selection(
    partitions: list[tuple[str, int, Path]],
    registry_sites: set[str],
) -> None:
    """Require exactly one monthly partition for every registry site."""

    discovered_sites = {site for site, _, _ in partitions}
    if discovered_sites != registry_sites:
        raise ValueError(
            "Legacy archive site names differ from the registry: missing={0}; extra={1}.".format(
                sorted(registry_sites - discovered_sites),
                sorted(discovered_sites - registry_sites),
            )
        )
    keys = {(site, month) for site, month, _ in partitions}
    if len(keys) != len(partitions):
        raise ValueError("The legacy archive contains duplicate site-month directories.")
    expected = {(site, month) for site in registry_sites for month in range(1, 13)}
    if keys != expected:
        raise ValueError(
            "The legacy archive does not contain exactly one partition for each site and month."
        )


def _expected_last_date(month: int, cutoff_date: date) -> date:
    """Return the required last full day for one calendar-month partition."""

    year = cutoff_date.year if month <= cutoff_date.month else cutoff_date.year - 1
    return date(year, month, _month_end(year, month))


def _validate_cutoff_coverage(
    coverage: PartitionCoverage,
    month: int,
    cutoff_date: date,
    start_year: int,
) -> None:
    """Require an uninterrupted history from ``start_year`` through the release cutoff."""

    if not coverage.years or coverage.years[0] != start_year:
        raise ValueError(
            "Normalized partition must begin in {0}; observed years begin in {1}.".format(
                start_year,
                coverage.years[0] if coverage.years else "none",
            )
        )
    expected_last = _expected_last_date(month, cutoff_date).isoformat()
    if coverage.last_date != expected_last:
        raise ValueError(
            "Normalized month {0:02d} ends at {1}; expected {2}.".format(
                month,
                coverage.last_date,
                expected_last,
            )
        )


def _check_new_outputs(daily_output: Path, alltimes_output: Path, report: Path) -> None:
    """Reject ambiguous output paths before any normalization work begins."""

    outputs = (daily_output, alltimes_output, report)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Daily output, all-times output, and report paths must be distinct.")
    for path in outputs:
        if path.exists():
            raise FileExistsError("Refusing to overwrite normalization output: {0}".format(path))


@contextmanager
def _normalization_lock(daily_output: Path):
    """Reserve a daily-output destination for one normalization invocation.

    The generated daily archive is the primary release artifact, so its parent
    directory provides a stable location for a lock shared by retries that
    target the same output. The lock is held across validation, staging, and
    publication; this prevents two long-running commands from racing at the
    final atomic rename.
    """

    daily_output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = daily_output.with_name(".{0}.normalization.lock".format(daily_output.name))
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            "Another normalization command is already using {0}. "
            "Wait for it to finish, or verify that it is stale before removing "
            "the lock file.".format(daily_output)
        ) from error

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("pid={0}\n".format(os.getpid()))
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _staging_path(destination: Path) -> Path:
    """Return a sibling staging path so publication is an atomic rename."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(".{0}.partial-{1}".format(destination.name, uuid4().hex))


def _daily_copies_match(alltimes_directory: Path, daily_directory: Path) -> bool:
    """Check that the runtime daily files equal their all-times archive copies."""

    return all(
        (alltimes_directory / filename).read_bytes()
        == (daily_directory / filename).read_bytes()
        for filename in _DAILY_FILENAMES
    )


def _normalization_report(
    *,
    input_root: Path,
    daily_output: Path,
    alltimes_output: Path,
    site_registry: Path,
    cutoff_date: date,
    partitions: list[NormalizedPartition],
) -> dict[str, object]:
    """Build an immutable, machine-readable account of a normalization run."""

    normalized = sorted(partitions, key=lambda item: (item.site, item.month))
    return {
        "normalization_schema_version": "1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_root": str(input_root),
        "daily_output": str(daily_output),
        "alltimes_output": str(alltimes_output),
        "site_registry": site_registry.name,
        "cutoff_date": cutoff_date.isoformat(),
        "validation": {
            "status": "passed",
            "checks": [
                "registry site membership",
                "complete site-month partition layout",
                "finite native weather records",
                "native three-hour timestamp completeness",
                "one finite daily record per native date",
                "complete calendar-month coverage through the release cutoff",
                "identical daily copies in both normalized archives",
            ],
        },
        "totals": {
            "partitions": len(normalized),
            "removed_native_records": sum(item.removed_native_records for item in normalized),
            "removed_daily_records": sum(item.removed_daily_records for item in normalized),
        },
        "partitions": [
            {
                "site": item.site,
                "month": item.month,
                "source": str(item.source.relative_to(input_root)),
                "first_date": item.coverage.first_date,
                "last_date": item.coverage.last_date,
                "years": list(item.coverage.years),
                "native_records": item.coverage.native_records,
                "daily_records": item.coverage.daily_records,
                "removed_native_records": item.removed_native_records,
                "removed_daily_records": item.removed_daily_records,
            }
            for item in normalized
        ],
    }


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize a legacy weather archive for a bounded release."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing 12-file SITE/MMMmm all-times legacy partitions.",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        required=True,
        help="New root for the normalized six-file legacy daily archive.",
    )
    parser.add_argument(
        "--alltimes-output",
        type=Path,
        required=True,
        help="New root for the normalized twelve-file daily plus native archive.",
    )
    parser.add_argument(
        "--site-registry",
        type=Path,
        required=True,
        help="CSV site registry defining the required complete site set.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=_parse_cutoff,
        required=True,
        help="Inclusive final date, which must be the end of a calendar month.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="New JSON report describing retained coverage and discarded rows.",
    )
    parser.add_argument(
        "--component-count",
        type=int,
        default=40,
        help="Number of stored PCA coefficients per atmospheric record (default: 40).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1980,
        help="Required first calendar year for every partition (default: 1980).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print each completed site-month partition.",
    )
    return parser


def _run_normalization(
    args: argparse.Namespace,
    *,
    input_root: Path,
    daily_output: Path,
    alltimes_output: Path,
    site_registry: Path,
    report: Path,
) -> int:
    """Build and atomically publish one fully validated normalized archive."""

    _check_new_outputs(daily_output, alltimes_output, report)
    partitions = _discover_partitions(input_root)
    _validate_partition_selection(partitions, _registry_sites(site_registry))
    staged_daily = _staging_path(daily_output)
    staged_alltimes = _staging_path(alltimes_output)
    staged_report = _staging_path(report)
    normalized: list[NormalizedPartition] = []

    try:
        for index, (site, month, source) in enumerate(partitions, 1):
            result: NormalizedLegacyPartition = normalize_legacy_partition(
                source,
                cutoff_date=args.cutoff_date,
                component_count=args.component_count,
            )
            coverage = validate_complete_calendar_coverage(result.partition)
            _validate_cutoff_coverage(
                coverage,
                month,
                args.cutoff_date,
                args.start_year,
            )

            label = source.name
            alltimes_directory = staged_alltimes / site / label
            daily_directory = staged_daily / site / label
            write_legacy_partition(
                alltimes_directory,
                result.partition,
                component_count=args.component_count,
                include_native=True,
            )
            write_legacy_partition(
                daily_directory,
                result.partition,
                component_count=args.component_count,
                include_native=False,
            )
            written = read_legacy_partition(
                alltimes_directory,
                component_count=args.component_count,
            )
            written_coverage = validate_complete_calendar_coverage(written)
            _validate_cutoff_coverage(
                written_coverage,
                month,
                args.cutoff_date,
                args.start_year,
            )
            if not _daily_copies_match(alltimes_directory, daily_directory):
                raise RuntimeError(
                    "Normalized daily copies do not match for {0}/{1}.".format(site, label)
                )
            normalized.append(
                NormalizedPartition(
                    site=site,
                    month=month,
                    source=source,
                    coverage=written_coverage,
                    removed_native_records=result.removed_native_records,
                    removed_daily_records=result.removed_daily_records,
                )
            )
            if args.progress:
                print(
                    "Normalized {0}/{1}: {2}/{3}".format(
                        index,
                        len(partitions),
                        site,
                        label,
                    ),
                    flush=True,
                )

        report_data = _normalization_report(
            input_root=input_root,
            daily_output=daily_output,
            alltimes_output=alltimes_output,
            site_registry=site_registry,
            cutoff_date=args.cutoff_date,
            partitions=normalized,
        )
        staged_report.write_text(
            json.dumps(report_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_daily.rename(daily_output)
        staged_alltimes.rename(alltimes_output)
        staged_report.rename(report)
    except Exception:
        shutil.rmtree(staged_daily, ignore_errors=True)
        shutil.rmtree(staged_alltimes, ignore_errors=True)
        staged_report.unlink(missing_ok=True)
        raise

    print("Wrote normalized daily archive to {0}".format(daily_output))
    print("Wrote normalized all-times archive to {0}".format(alltimes_output))
    print("Wrote normalization report to {0}".format(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Normalize a complete legacy archive and write a JSON provenance report."""

    parser = _arguments()
    args = parser.parse_args(argv)
    input_root = args.input_root.resolve()
    daily_output = args.daily_output.resolve()
    alltimes_output = args.alltimes_output.resolve()
    site_registry = args.site_registry.resolve()
    report = args.report.resolve()

    if not input_root.is_dir():
        parser.error("--input-root is not a directory: {0}".format(input_root))
    if not site_registry.is_file():
        parser.error("--site-registry is not a file: {0}".format(site_registry))
    if args.component_count <= 0:
        parser.error("--component-count must be positive.")
    if args.start_year > args.cutoff_date.year:
        parser.error("--start-year cannot be after --cutoff-date.")

    with _normalization_lock(daily_output):
        return _run_normalization(
            args,
            input_root=input_root,
            daily_output=daily_output,
            alltimes_output=alltimes_output,
            site_registry=site_registry,
            report=report,
        )


if __name__ == "__main__":
    sys.exit(main())
