# Legacy-to-Zarr Schema

Schema version 0.2.0 preserves the legacy PCA coefficients without changing
their numerical precision and adds precomputed physical native summaries.

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
    native_summary/{mean, median, good, bad}/
      {opacity, brightness_temperature, pwv_mm, wind_speed_m_s,
       surface_pressure_mbar, surface_temperature_k}
~~~

Each native date has exactly eight time_index values from 0 through 7,
representing UTC hours 0 through 21 in three-hour intervals. Daily records
remain explicitly stored because the legacy product averages reconstructed
physical spectra before recompressing them.

The `native_summary` arrays have eight rows, one per UTC time index. They
reduce the full site-month history in physical units: `mean` and `median` use
their corresponding NaN-aware reductions, while `good` and `bad` are the
15.87th and 84.13th percentiles. Opacity is reconstructed before reduction,
so the nonlinear PCA representation is not averaged directly.

The root attributes record the schema version, cadence, native-summary forms,
and derivation details.

## Release Manifest

Every release built with `ngehtsim-weather-import-legacy` has an adjacent
`<dataset>.zarr.manifest.json` file. The manifest is the immutable provenance
record for that particular artifact, and includes:

- the schema version, dataset identifier, builder revision, and build time;
- SHA-256 fingerprints of the frequency grid and all PCA arrays;
- the site-registry hash and fingerprints of both legacy PCA directories;
- SHA-256 hashes and sizes for every imported legacy binary file;
- per-site/month native and daily record counts, date range, years, and any
  explicitly removed malformed daily records; and
- the validation checks that completed before publication.

The manifest deliberately records source paths relative to `--legacy-root`,
never machine-specific absolute cluster paths.

## Historic Daily-Record Repair

The legacy postprocessor could write all-zero daily inputs for unavailable
days, producing non-finite opacity coefficients and duplicate dates. The
importer's `--repair-invalid-daily-records` option is an explicit migration
policy for this known defect. It drops only daily rows containing non-finite
values, then requires native/daily date equality and full calendar coverage.
It cannot manufacture missing weather data or make an incomplete partition
pass validation.
