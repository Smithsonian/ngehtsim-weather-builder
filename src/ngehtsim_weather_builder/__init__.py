"""Tools for building versioned ngehtsim weather datasets."""

from .dataset import PcaBasis, initialize_dataset, load_pca_basis, native_summaries, write_partition
from .importer import import_legacy_month
from .legacy import (
    LegacyFormatError,
    NormalizedLegacyPartition,
    WeatherPartition,
    WeatherRecords,
    normalize_legacy_partition,
    read_legacy_partition,
    validate_partition,
    write_legacy_partition,
)

__version__ = "0.3.0"

__all__ = [
    "LegacyFormatError",
    "NormalizedLegacyPartition",
    "PcaBasis",
    "WeatherPartition",
    "WeatherRecords",
    "import_legacy_month",
    "initialize_dataset",
    "load_pca_basis",
    "native_summaries",
    "normalize_legacy_partition",
    "read_legacy_partition",
    "validate_partition",
    "write_legacy_partition",
    "write_partition",
]
