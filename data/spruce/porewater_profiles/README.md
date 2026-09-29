# SPRUCE porewater-profile constraints

These files preserve the depth-profile constraints distributed with the
CLM-SPRUCE calibration workflow under `scripts/UQ/constraints_ch4/`:

- `CDOCS.txt`: dissolved organic carbon;
- `CACES.txt`: acetate; and
- `CCON_CH4S.txt`: dissolved methane.

Depth is in centimetres. DOC values are mmol C L-1 and are converted to
g C m-3 water with the carbon atomic mass. Acetate and methane are mmol L-1.
Uncertainties of `-999` mean unavailable. The standard ELM diagnostic uses
the 2013 observations by default, matching the profile-calibration workflow,
while retaining the other campaigns for sensitivity analyses.

The values were copied without modification from the CLM-SPRUCE repository.
Scientific provenance should be cited through the SPRUCE porewater datasets
and the associated Xu et al. and Ricciuto et al. publications; this directory
is a model-evaluation convenience copy, not a replacement data publication.
