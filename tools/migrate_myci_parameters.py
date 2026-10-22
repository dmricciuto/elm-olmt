#!/usr/bin/env python3
"""Add the published ELM-MYCI parameters to a current peatland parameter file.

The source file uses the original 25-PFT ordering.  The current peatland file
splits boreal spruce, tamarack, and shrub into upland and peatland PFTs and
adds moss and sedge.  Values are transferred by PFT name, never by index.
"""

import argparse
import shutil
from pathlib import Path

import netCDF4
import numpy as np


PFT_PARAMETERS = (
    "zwt_froot_a", "froot_radius", "froot_density", "vmax_froot_n",
    "vmax_froot_p", "km_froot_n", "km_froot_p", "inh_fungi_a",
    "inh_fungi_b", "inh_fungi_c", "vmax_fungi_din", "vmax_fungi_dip",
    "vmax_fungi_son", "vmax_fungi_sop",
)
GLOBAL_PARAMETERS = (
    "q10_upt", "swc_opt", "alpha_fpg", "km_nsc", "fungi_cost_n",
    "fungi_cost_p",
)
RENAMED_PFTS = {
    "upland_needleleaf_evergreen_boreal_tree": "needleleaf_evergreen_boreal_tree",
    "peatlnd_needleleaf_evergreen_boreal_tree": "needleleaf_evergreen_boreal_tree",
    "peatlnd_needleleaf_deciduous_boreal_tree": "needleleaf_deciduous_boreal_tree",
    "upland_broadleaf_deciduous_boreal_shrub": "broadleaf_deciduous_boreal_shrub",
    "peatlnd_broadleaf_deciduous_boreal_shrub": "broadleaf_deciduous_boreal_shrub",
}
ACTIVE_SOURCE_PFTS = (
    "needleleaf_evergreen_boreal_tree",
    "needleleaf_deciduous_boreal_tree",
    "broadleaf_deciduous_boreal_shrub",
)


def pft_names(nc):
    return [name.strip() for name in netCDF4.chartostring(nc["pftname"][:])]


def copy_variable(source, target, name, values):
    original = source[name]
    fill = getattr(original, "_FillValue", None)
    kwargs = {"fill_value": fill} if fill is not None else {}
    target_dimensions = tuple("allpfts" if dim == "allpft" else dim
                              for dim in original.dimensions)
    if any(dim not in target.dimensions for dim in target_dimensions):
        raise ValueError(f"No target dimensions for {name}: {target_dimensions}")
    created = target.createVariable(name, original.datatype, target_dimensions,
                                    **kwargs)
    for attr in original.ncattrs():
        if attr != "_FillValue":
            created.setncattr(attr, original.getncattr(attr))
    created[:] = values


def migrate(source_path, base_path, output_path):
    source_path = Path(source_path).resolve()
    base_path = Path(base_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path in (source_path, base_path):
        raise ValueError("Output must be distinct from both input files")
    if output_path.exists():
        raise FileExistsError(output_path)

    with netCDF4.Dataset(source_path) as source, netCDF4.Dataset(base_path) as base:
        names = pft_names(base)
        source_names = pft_names(source)
        source_index = {name: index for index, name in enumerate(source_names)}
        missing = [name for name in PFT_PARAMETERS + GLOBAL_PARAMETERS
                   if name not in source.variables]
        if missing:
            raise ValueError(f"Missing MYCI source variables: {missing}")
        if any(name in base.variables for name in PFT_PARAMETERS + GLOBAL_PARAMETERS):
            raise ValueError("Base parameter file already contains MYCI variables")
        if not set(ACTIVE_SOURCE_PFTS).issubset(source_index):
            raise ValueError("Source file lacks one or more MYCI vegetation types")
        mapping = []
        for name in names:
            source_name = RENAMED_PFTS.get(name, name)
            mapping.append(source_index.get(source_name))
        for name in ACTIVE_SOURCE_PFTS:
            if name not in [RENAMED_PFTS.get(target, target) for target in names]:
                raise ValueError(f"No target PFT for {name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_path, output_path)
    try:
        with netCDF4.Dataset(source_path) as source, netCDF4.Dataset(output_path, "a") as target:
            for name in PFT_PARAMETERS:
                original = source[name]
                if original.dimensions != ("pft",):
                    raise ValueError(f"Unexpected dimensions for {name}: {original.dimensions}")
                fill = getattr(original, "_FillValue", -1.0e20)
                values = np.full(len(names), fill, dtype=original.dtype)
                source_values = np.ma.filled(original[:], fill)
                for target_index, source_position in enumerate(mapping):
                    if source_position is not None:
                        values[target_index] = source_values[source_position]
                copy_variable(source, target, name, values)
            for name in GLOBAL_PARAMETERS:
                copy_variable(source, target, name, source[name][:])
            target.setncattr("myci_parameter_source", source_path.name)
            target.setncattr("myci_pft_mapping", "PFT-name mapping; split boreal types copied to upland and peatland variants")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(migrate(args.source, args.base, args.output))


if __name__ == "__main__":
    main()
