# Legacy Reference Scripts

These files are a source snapshot of the existing preprocessing workflow.
They are retained for scientific comparison while the new builder is developed.

`postproc_ray.py` uses `daily_postproc.py` to discover only complete daily
groups of eight three-hour inputs. Copy both files together when running the
legacy postprocessor outside this repository.

They require external MERRA-2 inputs, PCA bases, Ray/paramsurvey, and the
`am` radiative-transfer executable. Those dependencies and generated outputs
must not be committed to this repository.
