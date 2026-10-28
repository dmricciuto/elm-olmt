# SPRUCE ambient SP water-table screen (2026-10-28)

`config_files/SPRUCE_SP_ambient_WT_ensemble_20261028.cfg` runs 512 one-task
members with prescribed LAI/phenology and JULES peat hydraulics. Each member
has a 2006–2014 lead-in from the same mature restart, then a 2015–2023 ambient
evaluation. No warming treatments or Microbe nutrient feedbacks are active.
The fixed SPRUCE fen snow-retention fraction is 0.8; density targets and the
experimental partial-ice liquid pathway are not varied.

The smoke-tested executable was built from E3SM commit `e0622cf88b` plus
`ELM_SPWT_source_20261028.patch`. The SP moss-height fix to Pathfinder OLMT
is captured in `OLMT_SPWT_source_20261028.patch` against commit `113573c`.
These patches document uncommitted experiment code; they are not a claim that
the changes are ready for the main branches.

Nine parameters cover JULES retention (`sucsat`, `b`), saturated conductivity,
drainage strength and depth decay, ice impedance, and moss–soil water exchange.
Member 1 is the baseline, member 2 reproduces the earlier 32-member screen's
lowest thawed-day RMSE member, and the remaining 510 are a scrambled Sobol
sample. The parameter ranges and transforms are in
`generate_samples_20261028.py` and the accompanying metadata JSON.

On Pathfinder, use a login shell so CIME sees `MODULEPATH`:

```bash
cd /projects/hpcl-cli185/proj-shared/zdr/elm-olmt
/projects/hpcl-cli185/proj-shared/zdr/conda/envs/OLMT_pf/bin/python \
  elm_olmt.py --config config_files/SPRUCE_SP_ambient_WT_ensemble_20261028.cfg
```

`np_ensemble = 128` requests at most 128 simultaneous one-task members. On the
Pathfinder `parallel` partition, the generated script requests two exclusive
Big Loop nodes with 64 members per node, processed in four waves. The OLMT UQ
member histories are under
`/scratch/hpcl-cli185/zdr/e3sm_run/UQ/ensembles/`.

To rank completed ambient members (safe to rerun while others finish):

```bash
cd /projects/hpcl-cli185/proj-shared/zdr/elm-olmt
/projects/hpcl-cli185/proj-shared/zdr/conda/envs/OLMT_pf/bin/python \
  -m ensemble.spruce_sp_water_table.score_ambient_wt \
  --run-root /scratch/hpcl-cli185/zdr/e3sm_run/UQ/ensembles \
  --base-case SPRSPWT512_20261028_US-SPR_ICBELMBC \
  --observations data/spruce/spruce079/spruce_daily_water_table_processed_20250408.nc \
  --parm-list ensemble/spruce_sp_water_table/parm_list_SPRUCE_SP_ambient_WT_20261028.txt \
  --samples ensemble/spruce_sp_water_table/samples_SPRUCE_SP_ambient_WT_sobol512_20261028.txt \
  --output /scratch/hpcl-cli185/zdr/e3sm_run/UQ/analysis/SPRSPWT512_20261028_iceaware_scores.csv
```

The score combines thawed 2015–2020 water-table RMSE, seven-day-change RMSE,
and the 2021 growing-season low-water error. Frozen-period error and 2022–2023
holdout RMSE remain visible as separate columns. The ranking is a screening
criterion, not a posterior probability or a validated hydrologic calibration.

## Ice-free surrogate calibration

The weekly surrogate can be reused while restricting only the MCMC likelihood
to ice-free periods. Supply `--ice-free-reference-run` to
`calibrate_weekly_wt.py mcmc` and use a new output directory containing copies
of `surrogate_validation.json`, `weekly_surrogate.pkl`,
`weekly_observations.csv`, `weekly_model_m.npy`, and `samples_physical.npy` from
the original calibration. A week enters the likelihood only if all seven days
have `soilice_present = false` in the reference member's hollow column; the
threshold is 0.1 kg m⁻² total column ice by default. The mask is held fixed
across candidate parameter vectors because the surrogate has a fixed weekly
time axis. Frozen and partly frozen weeks remain in surrogate training but do
not contribute to this calibration or its temporal holdout score.

The 2026-10-28 experiment used member 381 as the reference ice mask, 2019–2021
for calibration, 2022–2023 for temporal holdout, and 60,000 MCMC steps. Its
Pathfinder results are under
`/scratch/hpcl-cli185/zdr/e3sm_run/UQ/analysis/SPRSPWT512_20261028_weekly_icefree_60k`.
