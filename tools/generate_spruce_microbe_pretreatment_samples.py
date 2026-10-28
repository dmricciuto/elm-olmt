#!/usr/bin/env python3
"""Generate the first-stage SPRUCE Microbe calibration design.

OLMT consumes physical parameter values, so this script applies linear or
log10 prior transforms before writing the ensemble sample file.  A second file
retains the underlying unit-cube coordinates for future surrogate construction
without requiring OLMT itself to understand log-space priors.

The design has 18 independent (latent) dimensions but 22 OLMT columns.  One
latent SOM-solubilization multiplier controls s1dom_f through s4dom_f, and one
latent mortality multiplier controls both methanogen death rates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Prior:
    name: str
    lower: float
    upper: float
    transform: str
    reference: float


PRIORS = (
    Prior("denitrification_multiplier", 0.03, 0.30, "log10", 0.10),
    Prior("moss_froot_leaf", 1.0, 3.5, "linear", 2.94680347971377),
    Prior("moss_hydraulic_conductivity_sat", 0.15, 15.0, "log10", 1.5),
    Prior("nh4_partition_coefficient", 0.001, 0.02, "log10", 0.005),
    Prior("decomp_depth_efolding", 0.25, 0.80, "linear", 0.395718086816327),
    Prior("dom_solubilization_anoxic_fraction", 0.0, 0.80, "linear", 0.50),
    Prior("l1dom_f", 0.08, 0.40, "linear", 0.20),
    Prior("som_dom_multiplier", 0.50, 2.0, "log10", 1.0),
    Prior("k_dom", 0.002, 0.020, "log10", 0.007004),
    Prior("saturated_macrodispersion", 2.0e-9, 1.0e-7, "log10", 5.0e-8),
    Prior("acetate_prod_max", 13.44, 215.04, "log10", 53.76),
    Prior("k_acetate", 4.0, 64.0, "log10", 16.0),
    Prior("acetate_methanogen_growth_rate", 0.084408, 0.337632, "log10", 0.168816),
    Prior("methanogen_mortality_multiplier", 0.40, 2.50, "log10", 1.0),
    Prior("k_aerobic_oxidation_ch4", 1.0, 100.0, "log10", 50.0),
    Prior("k_aerobic_oxidation_o2", 1.0, 80.0, "log10", 20.0),
    Prior("ch4_transport_threshold", 100.0, 1500.0, "log10", 750.0),
    Prior("plant_transport_coefficient", 0.002, 0.020, "log10", 0.007),
)


PHYSICAL_COLUMNS = (
    "microbe_methane_denitrification_rate_multiplier",
    "froot_leaf_pft18",
    "moss_water_hydraulic_conductivity_sat",
    "microbe_methane_aqueous_nh4_partition_coefficient",
    "decomp_depth_efolding",
    "dom_solubilization_anoxic_fraction",
    "l1dom_f",
    "s1dom_f",
    "s2dom_f",
    "s3dom_f",
    "s4dom_f",
    "k_dom_all_pfts",
    "microbe_methane_aqueous_dom_saturated_macrodispersion",
    "microbe_methane_acetate_prod_max",
    "microbe_methane_k_acetate",
    "microbe_methane_acetate_methanogen_growth_rate",
    "microbe_methane_acetate_methanogen_death_rate",
    "microbe_methane_h2_methanogen_death_rate",
    "microbe_methane_k_aerobic_oxidation_ch4",
    "microbe_methane_k_aerobic_oxidation_o2",
    "microbe_methane_ch4_transport_threshold",
    "microbe_methane_plant_transport_coefficient",
)


def transform(unit_samples: np.ndarray) -> np.ndarray:
    values = np.empty_like(unit_samples)
    for column, prior in enumerate(PRIORS):
        u = unit_samples[:, column]
        if prior.transform == "linear":
            values[:, column] = prior.lower + u * (prior.upper - prior.lower)
        elif prior.transform == "log10":
            lo = np.log10(prior.lower)
            hi = np.log10(prior.upper)
            values[:, column] = 10.0 ** (lo + u * (hi - lo))
        else:
            raise ValueError(f"unsupported transform: {prior.transform}")
    return values


def reference_unit_coordinates() -> np.ndarray:
    coordinates = np.empty(len(PRIORS))
    for column, prior in enumerate(PRIORS):
        if not prior.lower <= prior.reference <= prior.upper:
            raise ValueError(f"reference value outside prior for {prior.name}")
        if prior.transform == "linear":
            coordinates[column] = (
                (prior.reference - prior.lower) / (prior.upper - prior.lower)
            )
        else:
            coordinates[column] = (
                (np.log10(prior.reference) - np.log10(prior.lower))
                / (np.log10(prior.upper) - np.log10(prior.lower))
            )
    return coordinates


def expand_physical(latent: np.ndarray) -> np.ndarray:
    index = {prior.name: i for i, prior in enumerate(PRIORS)}
    col = lambda name: latent[:, index[name]]

    som_scale = col("som_dom_multiplier")
    mortality_scale = col("methanogen_mortality_multiplier")
    columns = (
        col("denitrification_multiplier"),
        col("moss_froot_leaf"),
        col("moss_hydraulic_conductivity_sat"),
        col("nh4_partition_coefficient"),
        col("decomp_depth_efolding"),
        col("dom_solubilization_anoxic_fraction"),
        col("l1dom_f"),
        0.18 * som_scale,
        0.14 * som_scale,
        0.10 * som_scale,
        0.06 * som_scale,
        col("k_dom"),
        col("saturated_macrodispersion"),
        col("acetate_prod_max"),
        col("k_acetate"),
        col("acetate_methanogen_growth_rate"),
        0.048 * mortality_scale,
        0.024 * mortality_scale,
        col("k_aerobic_oxidation_ch4"),
        col("k_aerobic_oxidation_o2"),
        col("ch4_transport_threshold"),
        col("plant_transport_coefficient"),
    )
    return np.column_stack(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=int, default=9, help="write 2**power members")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ensemble" / "spruce_pretreatment",
    )
    args = parser.parse_args()

    sampler = qmc.Sobol(d=len(PRIORS), scramble=True, seed=args.seed)
    unit_samples = sampler.random_base2(m=args.power)

    # Preserve a directly interpretable control member. Replacing one Sobol
    # point has negligible impact on the 512-member design and avoids spending
    # an additional run solely on the reference parameter set.
    unit_samples[0, :] = reference_unit_coordinates()
    latent = transform(unit_samples)
    physical = expand_physical(latent)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    member_count = physical.shape[0]
    stem = f"SPRUCE_microbe_pretreatment_sobol{member_count}_seed{args.seed}"
    physical_path = args.output_dir / f"samples_{stem}.txt"
    latent_path = args.output_dir / f"latent_unitcube_{stem}.txt"
    metadata_path = args.output_dir / f"metadata_{stem}.json"

    np.savetxt(
        physical_path,
        physical,
        fmt="%.12g",
        header=" ".join(PHYSICAL_COLUMNS),
    )
    np.savetxt(
        latent_path,
        unit_samples,
        fmt="%.12g",
        header=" ".join(prior.name for prior in PRIORS),
    )
    metadata = {
        "design": "scrambled Sobol with reference parameter set in member 1",
        "seed": args.seed,
        "members": member_count,
        "latent_dimensions": len(PRIORS),
        "physical_columns": list(PHYSICAL_COLUMNS),
        "priors": [prior.__dict__ for prior in PRIORS],
        "links": {
            "som_dom_multiplier": {
                "s1dom_f": 0.18,
                "s2dom_f": 0.14,
                "s3dom_f": 0.10,
                "s4dom_f": 0.06,
            },
            "methanogen_mortality_multiplier": {
                "microbe_methane_acetate_methanogen_death_rate": 0.048,
                "microbe_methane_h2_methanogen_death_rate": 0.024,
            },
        },
        "notes": [
            "OLMT reads the physical sample file directly; no surrogate or MCMC is required.",
            "Use the unit-cube file, not raw physical columns, for a future log-aware surrogate.",
            "Moss capillary nutrient access is disabled in the companion configuration.",
            "Peat bulk-density targets remain fixed at their parameter-file values.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(physical_path)
    print(latent_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
