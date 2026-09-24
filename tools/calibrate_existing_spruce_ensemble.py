#!/usr/bin/env python
"""Build a multi-treatment calibration from existing SPRUCE ensemble output."""

import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from sklearn.model_selection import train_test_split


TREATMENTS = ('TAMB', 'T0.00', 'T2.25', 'T4.50', 'T6.75', 'T9.00')
DEFAULT_CASE_TEMPLATE = '20260826_US-SPR_ICB20TRCNPRDCTCBC_{}_peatlandparms'
MONTH_DAYS = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
CALIBRATION_VARIABLES = (
    'AGNPP_pftgroup_tree',
    'AGNPP_pftgroup_shrub',
    'BGNPP_pftgroup_woody',
    'NPP_pftgroup_moss',
)
PFT_COMPONENTS = {
    'AGNPP_pftgroup_tree': (
        ('AGNPP', (25, 47), 0.36),
        ('AGNPP', (27, 49), 0.14),
    ),
    'AGNPP_pftgroup_shrub': (
        ('AGNPP', (36, 58), 0.25),
    ),
    'BGNPP_pftgroup_woody': (
        ('BGNPP', (25, 47), 0.36),
        ('BGNPP', (27, 49), 0.14),
        ('BGNPP', (36, 58), 0.25),
    ),
    'NPP_pftgroup_moss': (
        ('NPP', (40, 62), 0.25),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reuse portable postprocessed output for SPRUCE multi-treatment calibration.')
    parser.add_argument('--source-pkl-dir',
                        help='Directory containing copied, pre-run case pickles.')
    parser.add_argument('--case-template', default=DEFAULT_CASE_TEMPLATE,
                        help='Case-name template containing one treatment placeholder.')
    parser.add_argument('--ensemble-root',
                        help='Read-only root containing the existing treatment ensembles.')
    parser.add_argument('--obs-file',
                        help='SPRUCE treatment observation CSV.')
    parser.add_argument('--output-dir', required=True,
                        help='Writable directory for adapted pickles and calibration output.')
    parser.add_argument('--plots-only', action='store_true',
                        help='Regenerate plots from adapted pickles already in output-dir.')
    parser.add_argument('--mcmc-steps', type=int, default=0,
                        help='Run joint MCMC with this many steps after preparation.')
    parser.add_argument('--nwalkers', type=int, default=24)
    parser.add_argument('--fit-error', action='store_true',
                        help='Fit shared component error parameters during MCMC.')
    parser.add_argument('--exclude-zero-productivity', action='store_true',
                        help='Exclude nonpositive productivity samples independently by timestep.')
    parser.add_argument('--zero-productivity-threshold', type=float, default=0.0,
                        help='Minimum retained productivity when zero exclusion is enabled.')
    parser.add_argument('--reconstruct-samples-from-params', action='store_true',
                        help='Rebuild the sample matrix from each member parameter file.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.plots_only:
        missing = [name for name in ('source_pkl_dir', 'ensemble_root', 'obs_file')
                   if getattr(args, name) is None]
        if missing:
            parser.error('the following arguments are required unless --plots-only is used: '+
                         ', '.join('--'+name.replace('_', '-') for name in missing))
    return args


def case_name(treatment, case_template=DEFAULT_CASE_TEMPLATE):
    return case_template.format(treatment)


def array_checksum(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def annualize_daily_rate(monthly):
    """Convert monthly mean gC m-2 day-1 rates to annual gC m-2 yr-1 totals."""
    monthly = np.ma.asarray(monthly, dtype=float)
    if monthly.ndim != 2 or monthly.shape[0] % 12 != 0:
        raise ValueError('Expected a time-by-ensemble array with complete 12-month years; got '+
                         str(monthly.shape))
    nyears = monthly.shape[0] // 12
    reshaped = monthly.reshape(nyears, 12, monthly.shape[1])
    return np.ma.sum(reshaped * MONTH_DAYS.reshape(1, 12, 1), axis=1).filled(-9999.0)


def load_portable_output(path, expected_samples):
    with Dataset(path, 'r') as ds:
        frequency = str(getattr(ds, 'postprocessing_frequency', '')).lower()
        if frequency != 'monthly':
            raise ValueError(path+' has postprocessing_frequency='+frequency+', expected monthly')
        for variable in ('time', 'postprocessed'):
            if variable not in ds.variables:
                raise KeyError(variable+' not found in '+path)
        time = np.asarray(ds.variables['time'][:], dtype=float)
        if len(time) % 12 != 0:
            raise ValueError('Incomplete monthly time axis in '+path)
        years = np.asarray([int(round(value)) for value in time[::12]], dtype=float)
        output = {'taxis': years}
        annual = {}
        for components in PFT_COMPONENTS.values():
            for basevar, slots, _ in components:
                for slot in slots:
                    variable = basevar+'_pft'+str(slot)
                    if variable in annual:
                        continue
                    if variable not in ds.variables:
                        raise KeyError(variable+' not found in '+path)
                    data = ds.variables[variable][:]
                    if data.shape[1] != expected_samples:
                        raise ValueError(variable+' ensemble dimension mismatch in '+path)
                    annual[variable] = annualize_daily_rate(data)

        for target, components in PFT_COMPONENTS.items():
            values = np.zeros_like(next(iter(annual.values())))
            for basevar, slots, pft_fraction in components:
                paired_topounit_mean = 0.5 * (
                    annual[basevar+'_pft'+str(slots[0])] +
                    annual[basevar+'_pft'+str(slots[1])]
                )
                values = values + pft_fraction * paired_topounit_mean
            output[target] = values
        status = np.asarray(ds.variables['postprocessed'][:], dtype=int)
    return output, status


def _read_parameter_value(dataset, name, pft_index):
    if name not in dataset.variables:
        raise KeyError(name+' not found in '+dataset.filepath())
    values = np.ma.asarray(dataset.variables[name][:]).filled(np.nan)
    if values.ndim == 0 or values.size == 1:
        value = float(values.reshape(-1)[0])
    elif values.ndim == 1:
        value = float(values[int(pft_index)])
    else:
        raise ValueError(
            name+' has unsupported dimensions '+str(values.shape)+' in '+dataset.filepath())
    if not np.isfinite(value):
        raise ValueError(name+' is non-finite in '+dataset.filepath())
    return value


def _ensemble_member_dirs(case):
    member_dirs = sorted(
        path for path in glob.glob(os.path.join(case.rundir_UQ, 'g[0-9]*'))
        if os.path.isdir(path))
    if not member_dirs:
        raise FileNotFoundError('No ensemble member directories found in '+case.rundir_UQ)
    return member_dirs


def _read_member_sample(case, member_dir):
    parameter_files = sorted(glob.glob(os.path.join(member_dir, 'clm_params_*.nc')))
    if len(parameter_files) != 1:
        raise ValueError(
            'Expected one parameter file in '+member_dir+', found '+str(parameter_files))
    sample = np.empty(len(case.ensemble_parms), dtype=float)
    with Dataset(parameter_files[0], 'r') as dataset:
        for parameter_index, (name, pft_index) in enumerate(
                zip(case.ensemble_parms, case.ensemble_pfts)):
            sample[parameter_index] = _read_parameter_value(dataset, name, pft_index)
    return sample


def reconstruct_parameter_samples(case):
    """Recover the full sample matrix from authoritative member parameter files."""
    member_dirs = _ensemble_member_dirs(case)
    samples = np.empty((len(case.ensemble_parms), len(member_dirs)), dtype=float)
    for member_index, member_dir in enumerate(member_dirs):
        samples[:, member_index] = _read_member_sample(case, member_dir)

    saved_samples = np.asarray(case.samples, dtype=float)
    if saved_samples.shape[0] != samples.shape[0]:
        raise ValueError('Saved and reconstructed parameter counts differ for '+case.casename)
    overlap = min(saved_samples.shape[1], samples.shape[1])
    saved_design_matches = np.allclose(
        saved_samples[:, :overlap], samples[:, :overlap],
        rtol=2.0e-5, atol=1.0e-12)
    case.saved_design_matches_reconstructed = bool(saved_design_matches)
    if not saved_design_matches:
        difference = np.max(np.abs(saved_samples[:, :overlap] - samples[:, :overlap]))
        print(
            'Warning: the saved '+str(overlap)+'-sample design is stale for '+
            case.casename+'; using member parameter files (maximum absolute difference '+
            str(difference)+')', flush=True)
    print('Reconstructed '+str(samples.shape[1])+' samples for '+case.casename+
          ' from member parameter files', flush=True)
    return samples


def validate_reconstructed_sample_subset(case, samples):
    member_dirs = _ensemble_member_dirs(case)
    if len(member_dirs) != samples.shape[1]:
        raise ValueError(
            case.casename+' has '+str(len(member_dirs))+' members, expected '+
            str(samples.shape[1]))
    indices = sorted(set(np.linspace(0, len(member_dirs)-1, 11, dtype=int)))
    for member_index in indices:
        candidate = _read_member_sample(case, member_dirs[member_index])
        if not np.allclose(candidate, samples[:, member_index], rtol=2.0e-5, atol=1.0e-12):
            raise ValueError(
                'Treatment parameter sample mismatch for '+case.casename+
                ' member '+str(member_index+1))
    print('Verified '+str(len(indices))+' parameter samples for '+case.casename,
          flush=True)


def reset_surrogate_state(case):
    case.surrogate = {}
    case.pscaler = {}
    case.yscaler = {}
    case.svd_components = {}
    case.use_svd = {}
    case.surrogate_skipped = {}
    case.qoi_bad = {}
    case.qoi_bad_meanval = {}


def heldout_validation_scores(case, variable):
    """Reproduce the trainer's deterministic 80/20 validation scores by year."""
    values = np.asarray(case.output[variable], dtype=float).transpose()
    samples = np.asarray(case.samples, dtype=float).transpose()
    valid = np.where(values[:, 0] > -9999)[0]
    values = values[valid, :]
    samples = samples[valid, :]
    good_timesteps = [t for t in range(values.shape[1])
                      if t not in case.qoi_bad.get(variable, [])]
    if not good_timesteps:
        return []
    scores = []
    for timestep in good_timesteps:
        timestep_mask = np.isfinite(values[:, timestep]) & (values[:, timestep] > -9999)
        if bool(getattr(case, 'surrogate_exclude_zeros', False)):
            timestep_mask = timestep_mask & (
                values[:, timestep] > float(getattr(case, 'surrogate_zero_threshold', 0.0)))
        timestep_samples = samples[timestep_mask, :]
        timestep_values = values[timestep_mask, timestep]
        scaler = case.pscaler[variable][timestep]
        _, validation_samples, _, validation_values = train_test_split(
            scaler.transform(timestep_samples), timestep_values,
            test_size=0.2, random_state=42)
        prediction_scaled = case.surrogate[variable][timestep].predict(validation_samples)
        predictions = case.yscaler[variable][timestep].inverse_transform(
            prediction_scaled.reshape(-1, 1)).flatten()
        score = np.corrcoef(validation_values, predictions)[0, 1]**2
        scores.append((int(case.output['taxis'][timestep]), float(score)))
    return scores


def write_annual_validation_diagnostics(case):
    output_dir = os.path.join(case.UQ_output, 'surrogate')
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    plt.figure(figsize=(8, 5))
    for variable in CALIBRATION_VARIABLES:
        scores = heldout_validation_scores(case, variable)
        rows.extend((case.treatment_name, variable, year, score)
                    for year, score in scores)
        if scores:
            years, values = zip(*scores)
            plt.plot(years, values, marker='o', label=variable)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Year')
    plt.ylabel('Held-out validation R2')
    plt.title(case.treatment_name+' annual surrogate validation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'annual_validation_r2.png'), dpi=200)
    plt.close()
    with open(os.path.join(output_dir, 'annual_validation_r2.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['treatment', 'variable', 'year', 'validation_r2'])
        writer.writerows(rows)


def prepare_case(treatment, args, repo_root):
    name = case_name(treatment, args.case_template)
    source_pickle = os.path.join(args.source_pkl_dir, name+'.pkl')
    with open(source_pickle, 'rb') as handle:
        case = pickle.load(handle)

    if case.casename != name:
        raise ValueError('Case-name mismatch in '+source_pickle)
    case.OLMTdir = repo_root
    case.rundir_UQ = os.path.join(args.ensemble_root, name)
    case.UQ_output = os.path.join(args.output_dir, 'analysis', name)
    case.treatment_name = treatment
    case.postproc_freq = 'annual'
    case.postproc_startyear = 2015
    case.postproc_endyear = 2023
    case.postproc_vars = list(CALIBRATION_VARIABLES)
    case.postproc_pfts = []
    case.postproc_cols = []
    case.surrogate_exclude_zeros = bool(args.exclude_zero_productivity)
    case.surrogate_zero_threshold = float(args.zero_productivity_threshold)

    if args.reconstruct_samples_from_params:
        if not hasattr(args, '_reconstructed_samples'):
            args._reconstructed_samples = reconstruct_parameter_samples(case)
            args._saved_design_matches_reconstructed = bool(
                case.saved_design_matches_reconstructed)
        else:
            validate_reconstructed_sample_subset(case, args._reconstructed_samples)
            case.saved_design_matches_reconstructed = bool(
                args._saved_design_matches_reconstructed)
        case.samples = args._reconstructed_samples.copy()
        case.nsamples = case.samples.shape[1]
        case.nparms_ensemble = case.samples.shape[0]
        parameter_names = list(case.ensemble_parms)
        if 'vpd_min_moss' in parameter_names and 'vpd_max_moss' in parameter_names:
            lower = case.samples[parameter_names.index('vpd_min_moss'), :]
            upper = case.samples[parameter_names.index('vpd_max_moss'), :]
            ordered = int(np.sum(lower < upper))
            print(str(ordered)+'/'+str(case.nsamples)+
                  ' samples satisfy vpd_min_moss < vpd_max_moss', flush=True)

    portable_path = os.path.join(case.rundir_UQ, 'postprocessed_output.nc')
    case.output, case.postprocessed = load_portable_output(portable_path, case.nsamples)
    if not np.all(case.postprocessed == 1):
        raise ValueError('Not all members are successfully postprocessed for '+name)

    case.obs = {}
    case.obs_err = {}
    for variable in CALIBRATION_VARIABLES:
        case.get_fluxnet_obs(
            site='US-SPR', tstep='annual', ystart=2015, yend=2023,
            fluxnet_var=variable, myobsdir=args.obs_file, obs_format='table',
            error_fraction=0.15, error_floor=1.0)
    reset_surrogate_state(case)
    os.makedirs(case.UQ_output, exist_ok=True)
    trained = case.train_surrogate(CALIBRATION_VARIABLES)
    if set(trained) != set(CALIBRATION_VARIABLES):
        raise RuntimeError('Failed to train component surrogates for '+name+': '+str(trained))
    case.plot_surrogate(trained)
    write_annual_validation_diagnostics(case)
    return case


def validate_shared_ensemble(cases):
    reference = cases[TREATMENTS[0]]
    reference_checksum = array_checksum(reference.samples)
    metadata = (
        list(reference.ensemble_parms), list(reference.ensemble_pfts),
        list(reference.ensemble_pmin), list(reference.ensemble_pmax))
    for treatment in TREATMENTS[1:]:
        case = cases[treatment]
        if array_checksum(case.samples) != reference_checksum:
            raise ValueError('Parameter samples differ for treatment '+treatment)
        candidate = (
            list(case.ensemble_parms), list(case.ensemble_pfts),
            list(case.ensemble_pmin), list(case.ensemble_pmax))
        if candidate != metadata:
            raise ValueError('Parameter metadata differs for treatment '+treatment)
    return reference_checksum


def write_cases(cases, output_dir):
    pickle_dir = os.path.join(output_dir, 'pklfiles')
    os.makedirs(pickle_dir, exist_ok=True)
    treatment_cases = {treatment: cases[treatment].casename for treatment in TREATMENTS}
    launcher = cases[TREATMENTS[-1]]
    launcher.all_treatment_cases = treatment_cases
    for treatment, case in cases.items():
        destination = os.path.join(pickle_dir, case.casename+'.pkl')
        with open(destination, 'wb') as handle:
            pickle.dump(case, handle)
        print('Wrote '+destination, flush=True)
    return launcher


def regenerate_plots(output_dir, case_template):
    pickle_dir = os.path.join(output_dir, 'pklfiles')
    for treatment in TREATMENTS:
        path = os.path.join(pickle_dir, case_name(treatment, case_template)+'.pkl')
        with open(path, 'rb') as handle:
            case = pickle.load(handle)
        case.plot_surrogate(CALIBRATION_VARIABLES)
        write_annual_validation_diagnostics(case)
        print('Generated surrogate plots for '+treatment, flush=True)


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, repo_root)
    import model_ELM

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plots_only:
        regenerate_plots(args.output_dir, args.case_template)
        return

    args.source_pkl_dir = os.path.abspath(args.source_pkl_dir)
    args.ensemble_root = os.path.abspath(args.ensemble_root)
    args.obs_file = os.path.abspath(args.obs_file)
    np.random.seed(args.seed)

    cases = {}
    for treatment in TREATMENTS:
        print('Preparing '+treatment, flush=True)
        cases[treatment] = prepare_case(treatment, args, repo_root)

    sample_checksum = validate_shared_ensemble(cases)
    launcher = write_cases(cases, args.output_dir)
    manifest = {
        'treatments': list(TREATMENTS),
        'case_template': args.case_template,
        'cases': {treatment: cases[treatment].casename for treatment in TREATMENTS},
        'ensemble_root': args.ensemble_root,
        'source_pkl_dir': args.source_pkl_dir,
        'observation_file': args.obs_file,
        'variables': list(CALIBRATION_VARIABLES),
        'legacy_hr_excluded': True,
        'pft_component_method': 'configured PFT fractions times mean of hollow/hummock native PFT rates',
        'years': [2015, 2023],
        'nsamples': int(launcher.nsamples),
        'sample_sha256': sample_checksum,
        'mcmc_steps': int(args.mcmc_steps),
        'nwalkers': int(args.nwalkers),
        'fit_error': bool(args.fit_error),
        'exclude_zero_productivity': bool(args.exclude_zero_productivity),
        'zero_productivity_threshold': float(args.zero_productivity_threshold),
        'samples_reconstructed_from_member_parameter_files': bool(
            args.reconstruct_samples_from_params),
        'parameter_constraints': ['vpd_min_moss < vpd_max_moss'],
        'saved_pickle_design_matches_reconstructed': {
            treatment: bool(getattr(cases[treatment],
                                    'saved_design_matches_reconstructed', False))
            for treatment in TREATMENTS},
        'seed': int(args.seed),
    }
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    if args.mcmc_steps > 0:
        os.chdir(args.output_dir)
        launcher.MCMC(
            list(CALIBRATION_VARIABLES), nwalkers=args.nwalkers, nsteps=args.mcmc_steps,
            fit_error=args.fit_error, multitreatment=True)
        with open(os.path.join(args.output_dir, 'pklfiles', launcher.casename+'.pkl'), 'wb') as handle:
            pickle.dump(launcher, handle)
        print('Completed multi-treatment MCMC', flush=True)
    else:
        print('Prepared calibration objects; MCMC was not requested', flush=True)


if __name__ == '__main__':
    main()
