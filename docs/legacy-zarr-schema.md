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
Production releases will add immutable provenance fields such as the builder
commit, site-registry hash, PCA-basis hashes, and MERRA-2 coverage.
