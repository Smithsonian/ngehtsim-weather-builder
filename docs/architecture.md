# Weather Builder Architecture

## Scope

The builder transforms MERRA-2 inputs and `am` calculations into immutable,
versioned weather datasets for use by `ngehtsim`.

## Cadence

The native product contains eight UTC samples per day. The compatibility daily
product is derived by reconstructing the eight tau and Tb spectra, averaging
the physical spectra, and projecting the averages back onto the PCA basis.

Both native and daily products are retained in each dataset release.

Schema 0.2.0 also stores physical native summaries for each site-month and
three-hour UTC time index. These are the mean, median, 15.87th-percentile
(`good`), and 84.13th-percentile (`bad`) quantities across the historical
records. They avoid repeated PCA reconstruction during representative native
weather sampling while the raw native records remain available for exact-time
interpolation.

## Provenance and Validation

Each dataset release records the builder commit, site-registry hash, PCA-basis
hashes, MERRA-2 coverage, `am` version/configuration, frequency grid, units,
and content hashes.

The builder must reject incomplete site/date coverage. It must not silently
average fewer than eight native samples into a daily product.

## Data Distribution

Published Zarr datasets are immutable artifacts, not Git content and not
package data. `ngehtsim` will access them through a provider interface.

## Initial Migration

The first migration path imports the existing native alltimes and daily legacy
files. The importer validates their binary record lengths, coordinate
alignment, and complete native-day coverage before writing Zarr.
