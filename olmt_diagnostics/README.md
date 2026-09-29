# OLMT diagnostics

This package contains reusable readers, metrics, plots, and tabular products for
OLMT simulations.  Diagnostic products belong under `data/<site>/diagnostics/`
so that figures retain the processed values and provenance needed to recreate
them.

## SPRUCE treatment water table

Run the six standard non-eCO2 treatment comparison with:

```bash
python3 tools/run_spruce_diagnostics.py \
  --run-root /output/e3sm_run \
  --case-prefix 20260925_US-SPR_ICB20TRCNPRDCTCBC \
  --obs-file data/spruce/spruce079/spruce_daily_water_table_processed_20250408.nc \
  --output-dir data/spruce/diagnostics/20260925 \
  --start-year 2015 --end-year 2024
```

For the complete standard package, use `--full`:

```bash
python3 tools/run_spruce_diagnostics.py \
  --run-root /output/e3sm_run \
  --case-prefix CASEPREFIX_US-SPR_ICB20TRCNPRDCTCBC \
  --output-dir data/spruce/diagnostics/CASEPREFIX \
  --start-year 2015 --end-year 2023 \
  --profile-start-year 2019 --profile-end-year 2023 \
  --full
```

This writes a package manifest plus:

- hollow water-table comparisons and thawed-period statistics;
- tree and shrub AGNPP, woody BGNPP, moss NPP, and HR comparisons;
- treatment intercept and slope comparisons for those carbon-budget terms;
- hummock+hollow weighted DOC, acetate, and CH4 profiles; and
- the four-panel McFarlane peat-density, C:N, radiocarbon, and stock figure.

The model hollow water-table height is calculated as:

```text
water-table height above hollow (m) = -ZWT + H2OSFC / 1000
```

`ZWT` and `H2OSFC` must both be archived on a column-resolved daily history
tape.  Include `ZWT_col` and `H2OSFC_col` in the OLMT `[postprocessing]`
variables.  The reader discovers the appropriate history tape from variable
dimensions and does not assume a fixed `h1`, `h2`, or `h3` stream.

Each panel shades days when the layer-summed, topounit-weighted hollow-column
`SOILICE` inventory exceeds `0.1 kg m-2`. This small numerical threshold avoids
classifying trace roundoff as a frozen column and can be changed with
`--soilice-threshold-kg-m2`.

Panel annotations and the metrics CSV report the matched thawed days only:
model and observed means, ELM-minus-observed bias, RMSE, and squared Pearson
correlation (R-squared). Frozen periods remain visible as gray shading but do
not contribute to these statistics.

For legacy runs lacking column H2OSFC, `--missing-h2osfc zero` produces a
clearly labeled `-ZWT` approximation.  This mode must be requested explicitly;
it is not the default.

The command writes:

- the six-panel PNG;
- aligned daily model and observational values as CSV;
- per-treatment thawed-period means, bias, RMSE, and R-squared as CSV; and
- a JSON manifest recording inputs, dates, formula, and compatibility mode.

## SPRUCE carbon budget

Add `--include-carbon-budget` to generate comparisons against the processed
SPRUCE carbon-budget workbook. The standard configuration archives
`AGNPP_pft`, `BGNPP_pft`, `NPP_pft`, and `HR_col`. Model values are expressed
per hummock+hollow area; the unvegetated fen is excluded.

The diagnostic maps tree AGNPP to PFTs 3+5, shrub AGNPP to PFT 14, combined
tree+shrub BGNPP to PFTs 3+5+14, and moss NPP to PFT 18. It writes five
six-panel annual time-series comparisons and a sixth response page comparing
observed and modeled intercepts and slopes fitted across T0, T2.25, T4.50,
T6.75, and T9. The ambient enclosure is shown in the annual plots but excluded
from the temperature-response regression.

## SPRUCE peat profile and radiocarbon

Use `plot_spruce_mcfarlane_profiles.py` to compare a final ELM profile with the
2012 core observations used by McFarlane et al. (2018):

```bash
python3 olmt_diagnostics/plot_spruce_mcfarlane_profiles.py \
  '/output/e3sm_run/CASE_T0.00/run/CASE_T0.00.elm.h1.*.nc' \
  --output-dir data/spruce/diagnostics/CASE/mcfarlane2018
```

The four-panel product reports peat C density, C:N, bulk-peat Delta14C, and
cumulative peat C. Hummock and hollow model columns are shown separately and
as a topounit-weighted bog mean. Observations are summarized from the 16 treed
profiles and the two non-treed profiles used in the paper; their raw public
data and citation are retained under `data/spruce/mcfarlane2018/`.

Modeled peat C is the sum of the CWD, litter 1--3, SOM 1--4, DOM, bacteria,
and fungi fields available on the selected history tape. Missing components
are identified above the relevant subplot instead of being silently read from
a restart. The sampled cores retained vascular plant material, but ELM's root
C is not layer resolved. When `FROOTC`, `LIVECROOTC`, and `DEADCROOTC` are on
the history tape, the stock panel adds their whole-column total as a separate
marker. The paper's reported 176 +/- 40 kg C m-2 stock excludes raised-hummock
carbon, so the hollow result is the strictest stock comparison.

Radiocarbon from accelerated-decomposition spinup is an implementation and
equilibration diagnostic only. A quantitative comparison with the 2012
Delta14C observations requires physical-time final/transient simulation with
the historical atmospheric bomb curve.

When called through `run_spruce_diagnostics.py --full`, the McFarlane reader
uses the latest column/PFT-resolved T0 history file, falling back to TAMB and
then the base case. Override this with `--mcfarlane-state` when another stage
is scientifically preferable. Use `--mcfarlane-radiocarbon-note` to state
whether the run used a historical bomb curve, constant pre-bomb atmosphere,
or an experimental tracer.

## SPRUCE porewater profiles

The full package compares `MM_DOM_POREWATER_C`, `MM_ACETATE_C_SAT`, and
`MM_CH4_POREWATER` with the retained SPRUCE depth-profile constraints under
`data/spruce/porewater_profiles/`. Model profiles are area-weighted across the
hummock and hollow and exclude the auxiliary fen. The default model summary is
the late-growing-season distribution (DOY 121--260) for 2019--2023; both the
year range and day-of-year window are command-line options.

The acetate state is carbon mass per bulk layer volume. For comparison with
porewater measurements, the diagnostic divides by the layer liquid-water
fraction and by the mass of the two carbon atoms in acetate. The resulting
mol m-3 is numerically equal to mmol L-1. The plot shows treatment means; the
model CSV also retains daily 5th--95th percentile envelopes and sample counts.

## Shareable PDF report

After the full diagnostic package has been generated, assemble its figures
into a landscape report with:

```bash
python3 tools/build_spruce_diagnostics_report.py \
  --diagnostics-dir data/spruce/diagnostics/CASEPREFIX \
  --case-prefix CASEPREFIX_US-SPR_ICB20TRCNPRDCTCBC \
  --output output/pdf/ELM_Peatlands_Microbe_SPRUCE_Diagnostics.pdf
```

The report includes a run-provenance page followed by the McFarlane four-panel
comparison, water-table validation, five NPP/HR pages, treatment intercept and
slope diagnostics, and the porewater profile comparison. The PDF builder
fails if any required figure is absent rather than silently producing an
incomplete report.
