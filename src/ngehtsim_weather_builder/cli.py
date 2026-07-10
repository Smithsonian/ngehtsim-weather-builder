"""Command-line tools for building validated weather dataset releases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

import numpy as np

from .dataset import initialize_dataset, load_pca_basis, write_partition
from .legacy import LegacyFormatError, read_legacy_partition, repair_invalid_daily_records
from .manifest import ImportedPartition, build_manifest, write_manifest
from .validation import validate_complete_calendar_coverage, validate_written_partition


_MONTH_DIRECTORY = re.compile(
    r"^(?P<month>0[1-9]|1[0-2])(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$"
)


@dataclass(frozen=True)
class PartitionArgument:
    """One site-month legacy input selected on the command line."""

    site: str
    month: int
    source: Path


def parse_partition(value: str) -> PartitionArgument:
    """Parse SITE:MONTH:PATH without making path names shell-dependent."""

    fields = value.split(":", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("Partitions must use SITE:MONTH:PATH.")
    site, month_text, path_text = fields
    if not site:
        raise argparse.ArgumentTypeError("Partition site names cannot be empty.")
    try:
        month = int(month_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Partition month must be an integer.") from error
    if month < 1 or month > 12:
        raise argparse.ArgumentTypeError("Partition month must be from 1 through 12.")
    source = Path(path_text).resolve()
    if not source.is_dir():
        raise argparse.ArgumentTypeError("Partition directory does not exist: {0}".format(source))
    return PartitionArgument(site=site, month=month, source=source)


def discover_partitions(legacy_root: Path) -> list[PartitionArgument]:
    """Discover every SITE/MMMmm legacy partition below one archive root."""

    selections: list[PartitionArgument] = []
    for site_directory in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
        for month_directory in sorted(path for path in site_directory.iterdir() if path.is_dir()):
            match = _MONTH_DIRECTORY.fullmatch(month_directory.name)
            if match:
                selections.append(
                    PartitionArgument(
                        site=site_directory.name,
                        month=int(match.group("month")),
                        source=month_directory,
                    )
                )
    if not selections:
        raise ValueError("No legacy SITE/MMMmm directories were found below {0}.".format(legacy_root))
    return selections


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import validated legacy weather partitions into a Zarr dataset."
    )
    parser.add_argument("--output", required=True, type=Path, help="New output .zarr directory.")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Immutable identifier for this dataset release.",
    )
    parser.add_argument(
        "--builder-revision",
        required=True,
        help="Git commit of ngehtsim-weather-builder used for this build.",
    )
    parser.add_argument(
        "--legacy-root",
        required=True,
        type=Path,
        help="Root containing every selected legacy partition directory.",
    )
    parser.add_argument(
        "--site-registry",
        required=True,
        type=Path,
        help="CSV registry used to produce the selected weather data.",
    )
    parser.add_argument(
        "--tau-basis",
        required=True,
        type=Path,
        help="Legacy tau eigenspectra directory.",
    )
    parser.add_argument(
        "--tb-basis",
        required=True,
        type=Path,
        help="Legacy Tb eigenspectra directory.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--partition",
        action="append",
        type=parse_partition,
        metavar="SITE:MONTH:PATH",
        help="One legacy site-month directory; repeat for each partition.",
    )
    selection.add_argument(
        "--all-partitions",
        action="store_true",
        help="Discover and import every SITE/MMMmm directory below --legacy-root.",
    )
    parser.add_argument(
        "--repair-invalid-daily-records",
        action="store_true",
        help=(
            "Explicitly remove malformed legacy daily rows only when native and daily "
            "coverage validates afterward."
        ),
    )
    parser.add_argument("--component-count", type=int, default=40)
    parser.add_argument("--frequency-start-ghz", type=float, default=0.0)
    parser.add_argument("--frequency-stop-ghz", type=float, default=2000.0)
    return parser


def _check_new_output(output: Path, manifest_path: Path) -> None:
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing dataset: {0}".format(output))
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite existing manifest: {0}".format(manifest_path))


def _staging_paths(output: Path, manifest_path: Path) -> tuple[Path, Path]:
    suffix = uuid4().hex
    return (
        output.with_name(".{0}.partial-{1}".format(output.name, suffix)),
        manifest_path.with_name(".{0}.partial-{1}".format(manifest_path.name, suffix)),
    )


def main(argv: list[str] | None = None) -> int:
    """Build one release and write its adjacent JSON manifest."""

    parser = _arguments()
    args = parser.parse_args(argv)
    output = args.output.resolve()
    manifest_path = output.with_name("{0}.manifest.json".format(output.name))
    _check_new_output(output, manifest_path)

    legacy_root = args.legacy_root.resolve()
    if not legacy_root.is_dir():
        parser.error("--legacy-root is not a directory: {0}".format(legacy_root))
    for path, name in ((args.site_registry, "--site-registry"), (args.tau_basis, "--tau-basis"), (args.tb_basis, "--tb-basis")):
        if not path.exists():
            parser.error("{0} does not exist: {1}".format(name, path))

    selections = discover_partitions(legacy_root) if args.all_partitions else args.partition
    unique_keys = {(item.site, item.month) for item in selections}
    if len(unique_keys) != len(selections):
        parser.error("Each site-month partition may be specified only once.")

    basis = load_pca_basis(args.tau_basis, args.tb_basis, component_count=args.component_count)
    frequency_ghz = np.linspace(
        args.frequency_start_ghz,
        args.frequency_stop_ghz,
        basis.spectral_length,
        dtype=np.float64,
    )
    staged_output, staged_manifest = _staging_paths(output, manifest_path)
    imported: list[ImportedPartition] = []

    try:
        root = initialize_dataset(
            staged_output,
            basis,
            frequency_ghz,
            metadata={
                "dataset_id": args.dataset_id,
                "builder_revision": args.builder_revision,
            },
        )
        for selection in selections:
            try:
                if args.repair_invalid_daily_records:
                    repair = repair_invalid_daily_records(
                        selection.source,
                        component_count=basis.component_count,
                    )
                    partition = repair.partition
                    removed_daily_records = repair.removed_daily_records
                else:
                    partition = read_legacy_partition(
                        selection.source,
                        component_count=basis.component_count,
                    )
                    removed_daily_records = 0
            except LegacyFormatError as error:
                raise LegacyFormatError(
                    "Invalid legacy partition {0}/{1:02d} at {2}: {3}".format(
                        selection.site,
                        selection.month,
                        selection.source,
                        error,
                    )
                ) from error
            if not np.all(partition.native.month == selection.month):
                raise ValueError("Native records do not match requested month for {0}.".format(selection.site))
            if not np.all(partition.daily.month == selection.month):
                raise ValueError("Daily records do not match requested month for {0}.".format(selection.site))

            coverage = validate_complete_calendar_coverage(partition)
            write_partition(root, selection.site, selection.month, partition)
            validate_written_partition(root, selection.site, selection.month, partition)
            imported.append(
                ImportedPartition(
                    site=selection.site,
                    month=selection.month,
                    source=selection.source,
                    coverage=coverage,
                    removed_daily_records=removed_daily_records,
                )
            )

        removed_total = sum(item.removed_daily_records for item in imported)
        if removed_total:
            root.attrs["legacy_invalid_daily_records_removed"] = removed_total

        manifest = build_manifest(
            dataset_id=args.dataset_id,
            builder_revision=args.builder_revision,
            legacy_root=legacy_root,
            site_registry=args.site_registry,
            tau_basis_directory=args.tau_basis,
            tb_basis_directory=args.tb_basis,
            basis=basis,
            frequency_ghz=frequency_ghz,
            partitions=imported,
        )
        write_manifest(staged_manifest, manifest)
        staged_output.rename(output)
        staged_manifest.rename(manifest_path)
    except Exception:
        shutil.rmtree(staged_output, ignore_errors=True)
        staged_manifest.unlink(missing_ok=True)
        raise

    print("Wrote {0} validated partitions to {1}".format(len(imported), output))
    print("Wrote manifest to {0}".format(manifest_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
