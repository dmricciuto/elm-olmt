# SPRUCE soil-temperature seasonal benchmark

`seasonal_mean_2016_2023.csv` summarizes the non-elevated-CO₂ SPRUCE
experimental plots for the 2016–2023 treatment years. The source is
[SPRUCE.032, version 2.0](https://doi.org/10.3334/CDIAC/spruce.032), using
the `WEW_PLOT_XX_Complete_Environ_20260702.csv` files staged on Pathfinder at
`/projects/hpcl-cli185/world-shared/e3sm/inputdata/SPRUCE_data/2026_version`.
Raw half-hourly files are not copied into this repository.
`seasonal_mean_2023.csv` applies the same rules to 2023 alone for the
short-chain model year 1861, which uses 2023 meteorology under the current
2015–2023 CPL-bypass forcing cycle. It is a workflow comparison, not a
scientifically equilibrated spinup evaluation.

| Treatment | Plot | Warming target |
| --- | ---: | ---: |
| TAMB | 07 | Ambient outside enclosure |
| T0.00 | 06 | +0 °C enclosure |
| T2.25 | 20 | +2.25 °C |
| T4.50 | 13 | +4.50 °C |
| T6.75 | 08 | +6.75 °C |
| T9.00 | 17 | +9.00 °C |

The benchmark uses depth-labeled `TS_0` through `TS_200` sensors in the A, B,
and C arrays, at 0, 5, 10, 20, 30, 40, 50, 100, and 200 cm below the hollow
surface. Separate `TS_Hummock` probes are not mixed into these profiles.
`-9999`, nonnumeric records, and values outside -50 to 60 °C are excluded;
there is no gap filling. A probe contributes to a year/season/depth only if at
least 75% of the expected half-hourly observations are valid. Qualified probes
are averaged within each calendar year and season, then the qualified years are
averaged equally. Here DJF comprises December, January, and February of the
same calendar year; this is a multiyear seasonal climatology, not a continuous
single-winter trajectory. The CSV records the number of qualifying years and
probe-years so coverage can be audited.

Regenerate on Pathfinder with:

```bash
python olmt_diagnostics/spruce_soil_temperature_benchmark.py \
  --raw-dir /projects/hpcl-cli185/world-shared/e3sm/inputdata/SPRUCE_data/2026_version \
  --start-year 2016 --end-year 2023 \
  --summary data/spruce/soil_temperature_2026/seasonal_mean_2016_2023.csv \
  --figure output/soil_temperature/seasonal_profiles_2016_2023.png
```

To regenerate just the figure from the checked-in CSV, omit `--raw-dir` and
pass the same `--summary` and `--figure` paths. This does not need Pathfinder
or the large source files.

The current completed treatment simulation has only annual grid-mean `TSOI`
on h0. It cannot be compared with these seasonal hollow-referenced profiles.
The production `SPRUCE_microbe_20260924.cfg` now requests daily `TSOI_col` for
future runs. For a strict benchmark, compare model hollow column 1 with these
depth-referenced profiles, rather than the fen or a hummock/hollow area mean.
