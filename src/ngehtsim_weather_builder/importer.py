"""Convenience functions for importing one legacy site-month partition."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import zarr

from .dataset import PcaBasis, initialize_dataset, write_partition
from .legacy import read_legacy_partition


def import_legacy_month(
    legacy_directory: str | Path,
    output_path: str | Path,
    site: str,
    month: int,
    basis: PcaBasis,
    frequency_ghz: np.ndarray,
    metadata: Mapping[str, str] | None = None,
) -> zarr.Group:
    """Create a Zarr dataset containing one validated legacy site-month."""

    partition = read_legacy_partition(
        legacy_directory,
        component_count=basis.component_count,
    )
    root = initialize_dataset(output_path, basis, frequency_ghz, metadata=metadata)
    write_partition(root, site, month, partition)
    return root
