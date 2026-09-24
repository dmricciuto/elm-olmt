#!/usr/bin/env python
"""Summarize SPRUCE PFT output over the measured hollow/hummock area."""

import argparse
import os

import numpy as np
from netCDF4 import Dataset


TREATMENTS = ('TAMB', 'T0.00', 'T2.25', 'T4.50', 'T6.75', 'T9.00')
CASE_TEMPLATE = '20260826_US-SPR_ICB20TRCNPRDCTCBC_{}_peatlandparms'
MONTH_DAYS = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
MEASURED_HISTORY_TOPOUNITS = (2, 3)


def annual_flux_mean(variable):
    values = np.ma.asarray(variable[:], dtype=float)
    nyears = values.shape[0] // 12
    annual = np.ma.sum(
        values.reshape(nyears, 12, values.shape[1]) * MONTH_DAYS.reshape(1, 12, 1),
        axis=1)
    return float(np.ma.mean(annual))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble-root', required=True)
    args = parser.parse_args()

    pooled = {}
    treatment_totals = []
    print('treatment,pft_index,pft_name,measured_area_fraction,'
          'patch_npp_gC_m2_yr,measured_area_npp_contribution_gC_m2_yr,'
          'patch_totvegc_gC_m2,patch_totvegc_abg_gC_m2')

    for treatment in TREATMENTS:
        case = CASE_TEMPLATE.format(treatment)
        case_root = os.path.join(args.ensemble_root, case)
        portable_path = os.path.join(case_root, 'postprocessed_output.nc')
        history_path = os.path.join(
            case_root, 'g00001', case+'.elm.h2.2015-01-01-00000.nc')
        with Dataset(portable_path) as portable, Dataset(history_path) as history:
            slots = []
            for name, variable in portable.variables.items():
                if not name.startswith('NPP_pft'):
                    continue
                slot = int(getattr(variable, 'index'))
                topounit = int(history.variables['pfts1d_topounit'][slot])
                active = int(history.variables['pfts1d_active'][slot])
                weight = float(history.variables['pfts1d_wtgcell'][slot])
                if active <= 0 or weight <= 0 or topounit not in MEASURED_HISTORY_TOPOUNITS:
                    continue
                pft_index = int(getattr(variable, 'parameter_pft_index'))
                pft_name = str(getattr(variable, 'pft_name'))
                slots.append((slot, pft_index, pft_name, weight, variable))

            measured_weight = sum(item[3] for item in slots)
            if measured_weight <= 0:
                raise ValueError('No active measured-area PFTs found for '+case)

            grouped = {}
            for slot, pft_index, pft_name, weight, npp_variable in slots:
                group = grouped.setdefault(pft_index, {
                    'name': pft_name, 'weight': 0.0, 'npp_weighted': 0.0,
                    'vegc_weighted': 0.0, 'abg_weighted': 0.0})
                npp = annual_flux_mean(npp_variable)
                vegc = float(np.ma.mean(portable.variables['TOTVEGC_pft'+str(slot)][:]))
                abg = float(np.ma.mean(portable.variables['TOTVEGC_ABG_pft'+str(slot)][:]))
                group['weight'] += weight
                group['npp_weighted'] += weight*npp
                group['vegc_weighted'] += weight*vegc
                group['abg_weighted'] += weight*abg

            measured_npp = 0.0
            for pft_index in sorted(grouped):
                group = grouped[pft_index]
                area_fraction = group['weight']/measured_weight
                patch_npp = group['npp_weighted']/group['weight']
                contribution = group['npp_weighted']/measured_weight
                patch_vegc = group['vegc_weighted']/group['weight']
                patch_abg = group['abg_weighted']/group['weight']
                measured_npp += contribution
                print(','.join([
                    treatment, str(pft_index), group['name'], str(area_fraction),
                    str(patch_npp), str(contribution), str(patch_vegc), str(patch_abg)]))
                aggregate = pooled.setdefault(pft_index, {
                    'name': group['name'], 'area_fraction': 0.0, 'patch_npp': 0.0,
                    'contribution': 0.0, 'patch_vegc': 0.0, 'patch_abg': 0.0})
                aggregate['area_fraction'] += area_fraction
                aggregate['patch_npp'] += patch_npp
                aggregate['contribution'] += contribution
                aggregate['patch_vegc'] += patch_vegc
                aggregate['patch_abg'] += patch_abg

            gridcell_npp = annual_flux_mean(portable.variables['NPP'])
            treatment_totals.append((treatment, measured_weight, measured_npp, gridcell_npp))

    count = float(len(TREATMENTS))
    print('\npooled_mean,pft_index,pft_name,measured_area_fraction,'
          'patch_npp_gC_m2_yr,measured_area_npp_contribution_gC_m2_yr,'
          'patch_totvegc_gC_m2,patch_totvegc_abg_gC_m2')
    for pft_index in sorted(pooled):
        group = pooled[pft_index]
        print(','.join([
            'all_treatments', str(pft_index), group['name'],
            str(group['area_fraction']/count), str(group['patch_npp']/count),
            str(group['contribution']/count), str(group['patch_vegc']/count),
            str(group['patch_abg']/count)]))

    print('\ntreatment,measured_gridcell_fraction,measured_area_npp,archived_gridcell_npp')
    for row in treatment_totals:
        print(','.join(str(value) for value in row))


if __name__ == '__main__':
    main()
