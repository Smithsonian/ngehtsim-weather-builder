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
