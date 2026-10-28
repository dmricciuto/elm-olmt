"""Create a mixed peat/upland ELM parameter file from two scalar baselines.

The unqualified parameter keeps the historical upland value; ``*_peat``
preserves the current peatland calibration. Only parameters whose model
call sites select by topounit are migrated here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import netCDF4
import numpy as np


PARAMETERS = (
    "cn_s1",
    "cn_s2",
    "cn_s3",
    "cn_s4",
    "np_s1_new",
    "np_s2_new",
    "np_s3_new",
    "np_s4_new",
    "k_l3",
    "q10_hr",
    "decomp_depth_efolding",
    "mino2lim",
    "rsub_top_globalmax",
)


def create_parameter_file(peat_path: Path, upland_path: Path, output_path: Path) -> None:
    if output_path.resolve() in (peat_path.resolve(), upland_path.resolve()):
        raise ValueError("output must differ from both inputs")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    with netCDF4.Dataset(peat_path) as peat, netCDF4.Dataset(upland_path) as upland:
        for name in PARAMETERS:
            if name not in peat.variables or name not in upland.variables:
                raise ValueError(f"missing parameter: {name}")
            if peat[name].size != 1 or upland[name].size != 1:
                raise ValueError(f"incompatible dimensions for {name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(peat_path, output_path)
    with netCDF4.Dataset(upland_path) as upland, netCDF4.Dataset(output_path, "a") as mixed:
        for name in PARAMETERS:
            source = mixed.variables[name]
            old = upland.variables[name]
            peat_value = np.asarray(source[:]).copy()
            upland_value = np.asarray(old[:])
            peat_var = mixed.createVariable(name + "_peat", source.dtype, source.dimensions)
            peat_var[...] = peat_value.reshape(source.shape)
            if hasattr(source, "units"):
                peat_var.units = source.units
            peat_var.long_name = f"{name} on peat topounits; omitted means use {name}"
            source[...] = upland_value.reshape(source.shape)
            print(f"{name}: upland={upland_value.ravel()[0]:.8g}, "
                  f"peat={peat_value.ravel()[0]:.8g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peat", type=Path, required=True)
    parser.add_argument("--upland", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        create_parameter_file(args.peat, args.upland, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
