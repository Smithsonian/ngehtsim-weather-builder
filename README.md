# ngehtsim-weather-builder

Reproducible tooling for generating versioned atmospheric weather datasets used
by `ngehtsim`.

This repository contains source code, documentation, configuration examples,
and small test fixtures only. It must never contain MERRA-2 inputs, cluster
work products, `am` output, or published weather datasets.

The `legacy/` directory preserves the current preprocessing scripts as a
reference implementation. New modules will replace that workflow with a
validated pipeline that publishes immutable Zarr dataset releases.

## Validated legacy import

`ngehtsim-weather-import-legacy` creates a new Zarr release from explicitly
selected legacy site-month directories. It compares every written Zarr array
and dtype against the binary input, requires complete three-hour native days
and complete calendar-month coverage, then writes a sidecar JSON manifest.

The output Zarr directory and its adjacent `.manifest.json` file are immutable
release artifacts. The command refuses to overwrite either one.

The default import is strict and refuses malformed legacy records. The
historic archive currently contains a known daily-postprocessing defect. Use
`--repair-invalid-daily-records` only for that archive: it removes non-finite
daily rows, then proceeds only if the untouched native records prove that
complete, one-record-per-date daily coverage remains. Each removal is recorded
in the output Zarr attributes and manifest.

For example, this creates a small April validation release from the existing
legacy archive:

```bash
ngehtsim-weather-import-legacy \
  --output /path/to/published/legacy-april-sample.zarr \
  --dataset-id legacy-april-sample-v0.1.0 \
  --builder-revision "$(git rev-parse HEAD)" \
  --legacy-root /path/to/weather_data_alltimes \
  --site-registry legacy/Telescope_Site_Matrix.csv \
  --tau-basis /path/to/eigenspectra \
  --tb-basis /path/to/eigenspectra_Tb \
  --partition ALMA:4:/path/to/weather_data_alltimes/ALMA/04Apr
```

Replace the final `--partition` argument with `--all-partitions` to import a
complete legacy archive rooted at `--legacy-root`.

For the historic archive, add `--repair-invalid-daily-records` before
`--all-partitions`. Do not use this option to conceal incomplete or newly
generated inputs; those still fail validation.

For large imports, add `--progress` to report every completed site-month
partition with elapsed time and an estimate of the remaining time. Progress is
reported only after a partition has been written and validated.

Schema 0.2.0 additionally stores physical three-hour native summaries for
each site-month. The summaries match `ngehtsim`'s representative weather
forms (`mean`, `median`, `good`, and `bad`) and preserve raw native records
for exact-time interpolation.

Use an absolute output path outside either Git checkout. Published weather
datasets and their manifests must not be committed to this repository.

## Python support

Python 3.11 or newer is required.

## Development setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Repository boundaries

- `ngehtsim-weather-builder`: preprocessing and dataset-packaging source.
- Weather dataset releases: immutable artifacts stored outside Git.
- `ngehtsim`: runtime client and compatibility API.
