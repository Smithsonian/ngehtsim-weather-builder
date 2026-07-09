"""Tools for building versioned ngehtsim weather datasets."""

from .dataset import PcaBasis, initialize_dataset, load_pca_basis, write_partition
from .importer import import_legacy_month
from .legacy import (
    LegacyFormatError,
    WeatherPartition,
    WeatherRecords,
    read_legacy_partition,
    validate_partition,
)

__version__ = "0.1.0"

__all__ = [
    "LegacyFormatError",
    "PcaBasis",
    "WeatherPartition",
    "WeatherRecords",
    "import_legacy_month",
    "initialize_dataset",
    "load_pca_basis",
    "read_legacy_partition",
    "validate_partition",
    "write_partition",
]
