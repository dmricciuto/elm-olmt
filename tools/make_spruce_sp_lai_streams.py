#!/usr/bin/env python3
"""Build provisional 22-PFT SPRUCE LAI streams for an SP hydrology screen.

The surface dataset supplies monthly phenology.  Plot-level Sphagnum NPP is
used only as a bounded proxy for declining *effective* moss LAI; it is not a
measurement of moss leaf area.  Shrub LAI has an explicit warming sensitivity
because the local carbon-budget table does not contain measured shrub LAI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


TREATMENTS = {
    "TAMB": 0.0,
    "T0.00": 0.0,
    "T2.25": 2.25,
    "T4.50": 4.50,
    "T6.75": 6.75,
    "T9.00": 9.00,
}


def moss_factor(budget: pd.DataFrame, treatment: str, year: int) -> float:
    if treatment in ("TAMB", "T0.00") or year < 2016:
        return 1.0
    observations = budget[
        (budget.treatment == treatment) &
        (budget.model_var == "NPP_pftgroup_moss")
    ].set_index("year").obs
    control = budget[
        (budget.treatment == "T0.00") &
        (budget.model_var == "NPP_pftgroup_moss")
    ].set_index("year").obs
    if observations.empty or control.empty:
        raise ValueError(f"No Sphagnum NPP observations for {treatment}")
    # Hold the last measured year after 2021.  A cumulative minimum represents
    # persistent loss of cover, rather than a one-year productivity anomaly.
    years = [y for y in observations.index if 2016 <= y <= min(year, observations.index.max())]
    ratios = [max(0.0, float(observations.loc[y])) /
              max(1.0, float(control.loc[y])) for y in years]
    return float(np.clip(np.sqrt(min([1.0] + ratios)), 0.05, 1.0))


def write_streams(surface: Path, budget_file: Path, domain: Path,
                  output_dir: Path, shrub_gain_per_c: float,
                  shrub_peak_lai: float, moss_patch_lai: float) -> None:
    budget = pd.read_csv(budget_file)
    with xr.open_dataset(surface) as ds:
        monthly = ds.MONTHLY_LAI.isel(topounit=1, gridcell=0).values.copy()
    with xr.open_dataset(domain) as ds:
        lat = float(ds.yc.values[0, 0])
        lon = float(ds.xc.values[0, 0])
        area = float(ds.area.values[0, 0]) * 4.0 * np.pi * 6371.0**2
        mask = int(ds["mask"].values[0, 0])
    if monthly.shape[0] != 12 or monthly.shape[1] < 22:
        raise ValueError("Expected a 12-month, 22-PFT SPRUCE surface LAI template")
    if shrub_gain_per_c < 0 or shrub_peak_lai <= 0 or moss_patch_lai <= 0:
        raise ValueError("LAI anchors must be positive and shrub gain nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime([f"{y:04d}-{m:02d}-16"
                            for y in range(2006, 2024) for m in range(1, 13)])
    for treatment, warming in TREATMENTS.items():
        data = xr.Dataset(coords={"time": dates, "lat": [lat], "lon": [lon]})
        data["area"] = (("lat", "lon"), np.array([[area]], dtype=np.float32))
        data["mask"] = (("lat", "lon"), np.array([[mask]], dtype=np.int32))
        data["LATIXY"] = (("lat", "lon"), np.array([[lat]], dtype=np.float32))
        data["LONGXY"] = (("lat", "lon"), np.array([[lon]], dtype=np.float32))
        for pft in range(1, 22):
            values = np.tile(monthly[:, pft], 18).astype(np.float64)
            if pft == 14:  # Scale surface seasonality to an explicit peak anchor.
                values *= (shrub_peak_lai / monthly[:, 14].max()) * \
                          (1.0 + shrub_gain_per_c * warming)
            elif pft == 18:  # Moss: effective LAI declines with observed NPP.
                values = np.array([moss_patch_lai * moss_factor(budget, treatment, d.year)
                                   for d in dates], dtype=np.float64)
            data[f"LAI_{pft}"] = (("time", "lat", "lon"),
                                  np.maximum(values, 0.0)[:, None, None].astype(np.float32))
            data[f"LAI_{pft}"].attrs.update(long_name=f"prescribed PFT {pft} LAI",
                                              units="m2 leaf m-2 ground")
        data.attrs.update(
            title=f"Provisional SPRUCE SP LAI stream: {treatment}",
            note="Not a direct LAI observation. Moss effective LAI uses square-root "
                 "of the treatment/control Sphagnum NPP ratio and persistent decline; "
                 "the last measured moss year is held thereafter. Shrub LAI uses a "
                 "configurable warming gain. Tree and seasonal templates use the "
                 "SPRUCE surface dataset. LAI is shared across topounits for each PFT.",
            shrub_gain_per_C=shrub_gain_per_c,
            shrub_peak_lai_T0=shrub_peak_lai,
            moss_patch_lai_T0=moss_patch_lai,
            source_budget=str(budget_file),
            source_surface=str(surface),
        )
        target = output_dir / f"SPRUCE_SP_LAI_{treatment}.nc"
        data.to_netcdf(target, format="NETCDF4_CLASSIC", encoding={
            f"LAI_{pft}": {"zlib": True, "complevel": 1} for pft in range(1, 22)
        })
        print(target)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, default=root.parent /
                        "TES_SFA_ELMdata/sites/US-SPR/surfdata.nc")
    parser.add_argument("--domain", type=Path, default=root.parent /
                        "TES_SFA_ELMdata/sites/US-SPR/domain.nc")
    parser.add_argument("--budget", type=Path, default=root /
                        "data/spruce/spruce_c_budget_ambient_annual.csv")
    parser.add_argument("--output-dir", type=Path, default=root /
                        "data/spruce/lai_streams")
    parser.add_argument("--shrub-gain-per-c", type=float, default=0.04)
    parser.add_argument("--shrub-peak-lai", type=float, default=4.5)
    parser.add_argument("--moss-patch-lai", type=float, default=5.0)
    args = parser.parse_args()
    write_streams(args.surface, args.budget, args.domain,
                  args.output_dir, args.shrub_gain_per_c,
                  args.shrub_peak_lai, args.moss_patch_lai)


if __name__ == "__main__":
    main()
