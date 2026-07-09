# ngehtsim-weather-builder

Reproducible tooling for generating versioned atmospheric weather datasets used
by `ngehtsim`.

This repository contains source code, documentation, configuration examples,
and small test fixtures only. It must never contain MERRA-2 inputs, cluster
work products, `am` output, or published weather datasets.

The `legacy/` directory preserves the current preprocessing scripts as a
reference implementation. New modules will replace that workflow with a
validated pipeline that publishes immutable Zarr dataset releases.

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
