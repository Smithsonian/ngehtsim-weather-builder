# Legacy-to-Zarr Schema

Schema version 0.1.0 preserves the legacy PCA coefficients without changing
their numerical precision.

~~~text
/
  frequency_ghz
  pca/
    tau/{mean, components}
    tb/{mean, components}
  sites/{site}/months/{MM}/
    native/
      {year, day, time_index, tau_coefficients, tb_coefficients,
       pwv_mm, wind_speed_m_s, surface_pressure_mbar,
       surface_temperature_k}
    daily/
      {year, day, tau_coefficients, tb_coefficients, pwv_mm,
       wind_speed_m_s, surface_pressure_mbar, surface_temperature_k}
~~~

Each native date has exactly eight time_index values from 0 through 7,
representing UTC hours 0 through 21 in three-hour intervals. Daily records
remain explicitly stored because the legacy product averages reconstructed
physical spectra before recompressing them.

The root attributes record the schema version, cadence, and daily derivation.

## Release Manifest

Every release built with `ngehtsim-weather-import-legacy` has an adjacent
`<dataset>.zarr.manifest.json` file. The manifest is the immutable provenance
record for that particular artifact, and includes:

- the schema version, dataset identifier, builder revision, and build time;
- SHA-256 fingerprints of the frequency grid and all PCA arrays;
- the site-registry hash and fingerprints of both legacy PCA directories;
- SHA-256 hashes and sizes for every imported legacy binary file;
- per-site/month native and daily record counts, date range, and years; and
- the validation checks that completed before publication.

The manifest deliberately records source paths relative to `--legacy-root`,
never machine-specific absolute cluster paths.
