"""Generate the 512-member SPRUCE ambient SP hydrology design.

Member 1 is the current baseline and member 2 reproduces the best thawed-day
RMSE member from the 32-member screen. The remaining 510 rows are a seeded,
scrambled Sobol design. All sampled values are in native ELM parameter units.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
SEED = 20261028
NSAMPLES = 512
PARAMETERS = (
    # name, lower, upper, transform, baseline, previous best (screen member 7)
    ("peat_jules_sucsat_scale", 0.4, 2.5, "log", 1.0, 1.0),
    ("peat_jules_b_scale", 0.55, 1.6, "linear", 1.0, 1.0),
    ("peat_hydraulic_hksat_scale", 0.2, 5.0, "log", 1.0, 0.691580625407),
    ("rsub_top_globalmax_peat", 8.0e-7, 6.0e-5, "log", 2.87585115268938e-6, 1.37820887785e-6),
    ("peat_drainage_decay_depth", 0.2, 1.5, "linear", 0.4, 0.970077701844),
    ("soil_ice_impedance_exponent", 4.0, 12.0, "linear", 8.0, 11.8714884929),
    ("moss_water_hydraulic_conductivity_sat", 0.3, 5.0, "log", 1.5, 2.05604801605),
    ("moss_water_layer_thickness", 0.03, 0.10, "linear", 0.05, 0.090474523697),
    ("moss_water_substrate_rewetting_timescale_days", 0.3, 4.0, "log", 1.0, 0.619031788854),
)


def main() -> None:
    unit = qmc.Sobol(d=len(PARAMETERS), scramble=True, seed=SEED).random_base2(m=9)
    samples = np.empty_like(unit)
    for col, (_, lower, upper, transform, baseline, prior_best) in enumerate(PARAMETERS):
        if transform == "log":
            samples[:, col] = np.exp(np.log(lower) + unit[:, col] * np.log(upper / lower))
        else:
            samples[:, col] = lower + unit[:, col] * (upper - lower)
        if not lower <= baseline <= upper or not lower <= prior_best <= upper:
            raise ValueError(f"An anchor lies outside the range for {PARAMETERS[col][0]}")
        samples[0, col] = baseline
        samples[1, col] = prior_best

    parm_list = HERE / "parm_list_SPRUCE_SP_ambient_WT_20261028.txt"
    sample_file = HERE / "samples_SPRUCE_SP_ambient_WT_sobol512_20261028.txt"
    metadata_file = HERE / "metadata_SPRUCE_SP_ambient_WT_sobol512_20261028.json"
    parm_list.write_text(
        "# parameter PFT_index lower upper; index 0 denotes a scalar or first element\n"
        + "".join(f"{name} 0 {lower:.12g} {upper:.12g}\n"
                  for name, lower, upper, *_ in PARAMETERS)
    )
    np.savetxt(sample_file, samples, delimiter=",", fmt="%.12g")
    metadata_file.write_text(json.dumps({
        "purpose": "SPRUCE ambient hollow water-table calibration, SP mode",
        "seed": SEED,
        "members": NSAMPLES,
        "anchors": {"1": "current SP baseline", "2": "32-member screen member 7"},
        "parameter_order": [name for name, *_ in PARAMETERS],
        "transforms": {name: transform for name, _, _, transform, *_ in PARAMETERS},
        "notes": [
            "Peat density targets are fixed.",
            "The experimental partial-ice liquid pathway remains disabled.",
            "Run 2006-2014 lead-in, followed by 2015-2023 ambient evaluation.",
        ],
    }, indent=2) + "\n")
    assert samples.shape == (NSAMPLES, len(PARAMETERS))
    print(f"Wrote {NSAMPLES} samples for {len(PARAMETERS)} parameters")


if __name__ == "__main__":
    main()
