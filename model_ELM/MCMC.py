import numpy as np
from scipy.stats import norm
import model_surrogate as models
import os, math, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from optparse import OptionParser
import multiprocessing
import emcee
import time
import corner
import netCDF4 as nc
import shutil


ORDERED_PARAMETER_CONSTRAINTS = (
    ('vpd_min_moss', 'vpd_max_moss'),
)


def normalize_ordered_parameter_constraints(constraints):
    """Return ordered parameter constraints as ``(lower, upper)`` tuples."""
    if constraints is None:
        return ORDERED_PARAMETER_CONSTRAINTS
    if isinstance(constraints, str):
        constraints = [item.strip() for item in constraints.split(',') if item.strip()]
    normalized = []
    for constraint in constraints:
        if isinstance(constraint, str):
            if '<' not in constraint:
                raise ValueError('Ordered parameter constraint must use "<": '+constraint)
            lower_name, upper_name = [item.strip() for item in constraint.split('<', 1)]
        else:
            lower_name, upper_name = constraint
            lower_name = str(lower_name).strip()
            upper_name = str(upper_name).strip()
        if not lower_name or not upper_name:
            raise ValueError('Ordered parameter constraint has an empty parameter name')
        normalized.append((lower_name, upper_name))
    return tuple(normalized)


def parameters_satisfy_constraints(parms, parameter_names,
        ordered_constraints=ORDERED_PARAMETER_CONSTRAINTS):
    """Return whether named model parameters satisfy physical ordering rules."""
    if parameter_names is None:
        return True
    parameter_names = list(parameter_names)
    for lower_name, upper_name in normalize_ordered_parameter_constraints(ordered_constraints):
        if lower_name not in parameter_names or upper_name not in parameter_names:
            continue
        lower = parms[parameter_names.index(lower_name)]
        upper = parms[parameter_names.index(upper_name)]
        if not lower < upper:
            return False
    return True


def sample_from_prior(pmin, pmax, nsamples, parameter_names=None,
        ordered_constraints=ORDERED_PARAMETER_CONSTRAINTS):
    nparms = len(pmin)
    #Uniform priors
    samples = []
    while len(samples) < nsamples:
        candidates = np.random.uniform(
            low=np.array(pmin), high=np.array(pmax),
            size=(max(nsamples - len(samples), 16), nparms))
        samples.extend(candidate for candidate in candidates
                       if parameters_satisfy_constraints(
                           candidate, parameter_names, ordered_constraints))
    return np.asarray(samples[:nsamples])


def log_posterior(parms, targets, myvars, pmin, pmax, obs, obs_err,
        nparms_ensemble, nerr_parms, run_surrogate, parameter_names,
        ordered_constraints=ORDERED_PARAMETER_CONSTRAINTS):
    # Quick prior check first (fastest rejection)
    if np.any(parms < pmin) or np.any(parms > pmax):
        return -np.inf  # Use -inf instead of -9999999 (emcee standard)
    if not parameters_satisfy_constraints(parms, parameter_names, ordered_constraints):
        return -np.inf
    
    log_likelihood = 0.0
    parms_model = parms[:(nparms_ensemble - nerr_parms)]
    
    try:
        for target in targets:
            if target not in run_surrogate:
                continue
                
            # Get model predictions for this target
            output = run_surrogate[target](parms_model.reshape(1, -1), myvars)
            
            for i, v in enumerate(myvars):
                if v not in output or v not in obs[target] or v not in obs_err[target]:
                    continue
                myoutput = output[v].flatten()
                myobs = obs[target][v]
                myerr = obs_err[target][v]
                if len(myoutput) != len(myobs) or len(myobs) != len(myerr):
                    return -np.inf
                
                # Vectorized masking
                mask = (myobs > -9000) & (myerr > 0)
                if not np.any(mask):
                    continue
                
                # Apply mask once
                obs_masked = myobs[mask]
                output_masked = myoutput[mask]
                err_masked = myerr[mask]
                
                # Update error if fitting error parameters
                if nerr_parms > 0:
                    err_masked = np.full_like(err_masked, parms[nparms_ensemble - nerr_parms + i])
                
                # Vectorized likelihood calculation
                residuals = output_masked - obs_masked
                log_likelihood += -0.5 * np.sum(
                    np.log(2 * np.pi * err_masked**2) + (residuals / err_masked)**2
                )
                
    except Exception as e:
        return -np.inf
    
    return log_likelihood

# More sophisticated burn-in detection
def estimate_burnin(sampler, labels_model):
    """Estimate burn-in period using autocorrelation time"""
    try:
        # Get autocorrelation time
        tau = sampler.get_autocorr_time(quiet=True)
        max_tau = np.max(tau)
        
        # Rule of thumb: burn-in = 2 * max(autocorr_time)
        burnin = int(2 * max_tau)
        
        print(f"Autocorrelation times: {tau}")
        print(f"Estimated burn-in: {burnin} steps")
        
        return max(burnin, sampler.chain.shape[0] // 10)  # At least 10%
        
    except Exception as e:
        print(f"Could not estimate autocorr time: {e}")
        return sampler.chain.shape[0] // 3  # Fall back to 33%

#-------------------------------- MCMC ------------------------------------------------------

def _mcmc_treatment_items(self):
    treatment_cases = getattr(self, 'all_treatment_cases', {})
    if isinstance(treatment_cases, dict):
        return list(treatment_cases.items())
    if isinstance(treatment_cases, (list, tuple)):
        return [(str(case_name), str(case_name)) for case_name in treatment_cases]
    return []


def MCMC(self, myvars, nwalkers=32, nsteps=100, fit_error=True, multisite=False,
        multitreatment=False):
    if multisite and multitreatment:
        raise ValueError('MCMC can run multisite or multitreatment mode, not both at once')

    target_kind = 'site'
    if multitreatment:
        target_kind = 'treatment'
        target_items = _mcmc_treatment_items(self)
        if len(target_items) == 0:
            treatment = getattr(self, 'treatment_name', self.casename)
            target_items = [(treatment, self.casename)]
    elif multisite and hasattr(self, 'all_sites') and self.all_sites is not None:
        target_items = [(s, self.casename.replace(self.site, s)) for s in self.all_sites]
    else:
        target_items = [(self.site, self.casename)]
    targets = [target for target, _ in target_items]
    
    pmin, pmax, nparms_ensemble = self.ensemble_pmin, self.ensemble_pmax, \
        self.nparms_ensemble
    run_surrogate = {}
    obs = {}
    obs_err = {}
    thiscase={}
    
    for target, case_name in target_items:
        if case_name == self.casename:
            obs[target] = self.obs.copy()
            obs_err[target] = self.obs_err.copy()
            run_surrogate[target] = self.run_surrogate
        else:
            from model_ELM import ELMcase
            # Get the case objects for other sites/treatments.
            thiscase[target] = ELMcase(casename=case_name)
            run_surrogate[target] = thiscase[target].run_surrogate
            obs[target] = thiscase[target].obs.copy()
            obs_err[target] = thiscase[target].obs_err.copy()

    #Add parameters to estimate observation error stddev
    nerr_parms = 0    
    ensemble_parms = self.ensemble_parms.copy()
    if (fit_error):
        print("Fitting observation error parameters")
        for v in myvars:
            # Use all targets to set a shared error-prior scale for this variable.
            max_obs = 0.01
            for target in targets:
                if v not in obs[target] or v not in obs_err[target]:
                    continue
                mask = (obs[target][v] > -9000) & (obs_err[target][v] > 0)
                if np.any(mask):
                    max_obs = max(max_obs, np.max(np.abs(obs[target][v][mask])))
            err_prior_min = 0.0
            err_prior_max = 0.25 * max_obs

            # Add error parameter to prior ranges and names
            pmin = np.append(pmin, err_prior_min)
            pmax = np.append(pmax, err_prior_max)
            ensemble_parms = ensemble_parms + ['sigma_'+v]
            nparms_ensemble = len(ensemble_parms)
            nerr_parms = nerr_parms+1

    # Initialize walkers in the prior space
    ordered_constraints = normalize_ordered_parameter_constraints(
        getattr(self, 'calibration_parameter_constraints', None))
    active_constraints = [
        lower+' < '+upper for lower, upper in ordered_constraints
        if lower in ensemble_parms and upper in ensemble_parms]
    if active_constraints:
        print('Enforcing parameter constraints: '+', '.join(active_constraints))
    p0 = sample_from_prior(
        pmin, pmax, nwalkers, ensemble_parms, ordered_constraints)

    # Set up the sampler and run MCMC
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            nparms_ensemble,
            log_posterior,
            args=(targets, myvars, pmin, pmax, obs, obs_err, nparms_ensemble,
                  nerr_parms, run_surrogate, ensemble_parms, ordered_constraints),
            #pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    #Get the samples, likelihoods and best parameters
    n_model_parms = len(ensemble_parms) - nerr_parms

    labels_model = ensemble_parms[:n_model_parms]
    burnin = estimate_burnin(sampler, labels_model)
    print(f"Using burn-in of {burnin} steps out of {nsteps} total steps ({burnin/nsteps*100:.1f}%)")
    
    samples = sampler.get_chain(discard=burnin, thin=5, flat=True)
    log_probs = sampler.get_log_prob(discard=burnin, thin=5, flat=True)
    best_idx = np.argmax(log_probs)
    best_parms = samples[best_idx, :n_model_parms]
    print("Mean of each parameter:")
    print(np.mean(samples, axis=0))
    print("Standard deviation of each parameter:")
    print(np.std(samples, axis=0))
    print("Best-fit parameters:")
    print(best_parms)
    if (fit_error):
        best_err_parms = samples[best_idx, n_model_parms:]
        print("Best-fit error parameters:")
        print(best_err_parms)

    # Plot histograms for each parameter
    MCMC_out = self.UQ_output + '/MCMC_output/'
    if (multisite):
        MCMC_out = MCMC_out+'/multisite/'
    if (multitreatment):
        MCMC_out = MCMC_out+'/multitreatment/'
    outdir = MCMC_out+'/plots/pdfs'
    os.makedirs(outdir, exist_ok=True)
    for i in range(samples.shape[1]):
        plt.figure()
        plt.hist(samples[:, i], bins=25, density=True, alpha=0.7)
        plt.xlabel(ensemble_parms[i])
        plt.ylabel('Probability Density')
        plt.title(f'Posterior of {ensemble_parms[i]}')
        plt.savefig(f'{outdir}/{ensemble_parms[i]}'+'.png')
        plt.close()

    n_samples = samples.shape[0]
    outputs_by_target = {}
    default_outputs_by_target = {}
    for target in targets:
        output_dict = {v: [] for v in myvars}
        default_output_dict = {v: [] for v in myvars}  # ADD THIS: Store default predictions
        
        # Run MCMC samples through surrogate
        for i in range(n_samples):
            parms_model = samples[i, :nparms_ensemble - nerr_parms]
            output = run_surrogate[target](parms_model.reshape(1, -1), myvars)
            for v in myvars:
                output_dict[v].append(output[v].flatten())

        # Convert lists to arrays
        for v in myvars:
            output_dict[v] = np.array(output_dict[v])  # shape: (n_samples, n_obs)

        #Run default parameters through surrogate model
        if hasattr(self, 'default_parms') and self.default_parms:
            print(f"Running default parameters through surrogate model for {target_kind} {target}")
            try:
                # Convert default_parms to the format expected by surrogate
                default_parms_array = np.array(self.default_parms).reshape(1, -1)
                default_output = run_surrogate[target](default_parms_array, myvars)
                # Store default predictions
                for v in myvars:
                    default_output_dict[v] = default_output[v].flatten()
            except Exception as e:
                print(f"Error running default parameters through surrogate for {target_kind} {target}: {e}")
                # Set to None to indicate failure
                for v in myvars:
                    default_output_dict[v] = None
        else:
            for v in myvars:
                default_output_dict[v] = None
        outputs_by_target[target] = output_dict
        default_outputs_by_target[target] = default_output_dict

        # Plot predictions with 95% confidence intervals AND default parameters
        outdir_pred = MCMC_out + 'plots/predictions/'
        if (multisite or multitreatment):
            outdir_pred = os.path.join(outdir_pred, str(target).replace(os.sep, '_'))
        os.makedirs(outdir_pred, exist_ok=True)
        
        for v in myvars:
            # Compute percentiles from MCMC samples
            lower = np.percentile(output_dict[v], 2.5, axis=0)
            upper = np.percentile(output_dict[v], 97.5, axis=0)
            median = np.percentile(output_dict[v], 50, axis=0)
            x = np.arange(len(median))
            
            # Prepare observations
            obs_plot = np.array(obs[target][v].copy())
            obs_plot[obs_plot < -9000] = np.nan
            obs_err_plot = np.array(obs_err[target][v].copy())
            obs_err_plot[obs_err_plot < -9000] = np.nan
            
            if (fit_error):
                err_idx = ensemble_parms.index('sigma_'+v)
                obs_err_plot[obs_err_plot > -9000] = best_err_parms[err_idx - n_model_parms]

            # Create the plot
            plt.figure(figsize=(12, 6))

            # Plot MCMC uncertainty
            plt.fill_between(x, lower, upper, color='gray', alpha=0.5, label='95% CI (MCMC)')
            plt.plot(x, median, 'r-', linewidth=3, label='Model median (MCMC)')

            #Plot default parameters if available 
            if default_output_dict[v] is not None:
                plt.plot(x, default_output_dict[v], 'k-', linewidth=3, 
                        label='Default parameters', alpha=0.9)

            # Plot observations 
            plt.errorbar(x, obs_plot, yerr=obs_err_plot, fmt='bo', 
                        label='Observations', alpha=0.8, markersize=6, linewidth=2)

            plt.xlabel('Time', fontsize=14)
            plt.ylabel(v, fontsize=14)
            plt.title(f'Posterior predictive for {v} ({target_kind.title()}: {target})', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{outdir_pred}/Predictions_{v}_posterior.png', dpi=300, bbox_inches='tight')
            plt.close()

            #Create a separate plot showing residuals 
            if default_output_dict[v] is not None:
                plt.figure(figsize=(12, 4))
                
                # Calculate residuals
                obs_valid = obs_plot[~np.isnan(obs_plot)]
                x_valid = x[~np.isnan(obs_plot)]
                default_valid = default_output_dict[v][~np.isnan(obs_plot)]
                median_valid = median[~np.isnan(obs_plot)]
                
                if len(obs_valid) > 0:
                    default_residuals = default_valid - obs_valid
                    mcmc_residuals = median_valid - obs_valid
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(x_valid, default_residuals, 'k-', linewidth=3, label='Default residuals')
                    plt.plot(x_valid, mcmc_residuals, 'r-', linewidth=3, label='MCMC median residuals')
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
                    plt.xlabel('Time', fontsize=12)
                    plt.ylabel('Model - Obs', fontsize=12)
                    plt.title(f'Residuals for {v}', fontsize=14)
                    plt.legend(fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(fontsize=11)
                    plt.yticks(fontsize=11)
                    
                    plt.subplot(1, 2, 2)
                    plt.hist(default_residuals, bins=20, alpha=0.7, label='Default', color='black', linewidth=2)
                    plt.hist(mcmc_residuals, bins=20, alpha=0.7, label='MCMC median', color='red', linewidth=2)
                    plt.xlabel('Model - Obs', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.title(f'Residual Distribution for {v}', fontsize=14)
                    plt.legend(fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(fontsize=11)
                    plt.yticks(fontsize=11)
                    
                    plt.tight_layout()
                    plt.savefig(f'{outdir_pred}/Residuals_{v}_comparison.png', dpi=300, bbox_inches='tight')
                    plt.close()

    # Write summary statistics to file
    summary_file = MCMC_out + '/prediction_summary_stats.txt'
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PREDICTION SUMMARY STATISTICS\n")
        f.write("="*60 + "\n")
        f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of MCMC samples: {n_samples}\n")
        f.write(f"Burn-in: {burnin} steps ({burnin/nsteps*100:.1f}%)\n")
        if multisite:
            f.write(f"Multi-site analysis with {len(targets)} sites: {targets}\n")
        elif multitreatment:
            f.write(f"Multi-treatment analysis with {len(targets)} treatments: {targets}\n")
        else:
            f.write(f"Single-site analysis: {targets[0]}\n")
        f.write("\n")
        
        # Track overall statistics
        all_default_rmse = []
        all_mcmc_rmse = []
        all_improvements = []
        
        for target in targets:
            output_dict = outputs_by_target[target]
            default_output_dict = default_outputs_by_target[target]
            f.write(f"\n{target_kind.upper()}: {target}\n")
            f.write("-" * 40 + "\n")
            
            site_default_rmse = []
            site_mcmc_rmse = []
            site_improvements = []
            
            for v in myvars:
                f.write(f"\nVariable: {v}\n")
                
                # Get valid observations for this target/variable
                obs_plot = np.array(obs[target][v].copy())
                obs_plot[obs_plot < -9000] = np.nan
                obs_valid = obs_plot[~np.isnan(obs_plot)]
                
                if len(obs_valid) > 0 and default_output_dict[v] is not None:
                    # Get predictions for this target/variable
                    lower = np.percentile(output_dict[v], 2.5, axis=0)
                    upper = np.percentile(output_dict[v], 97.5, axis=0)
                    median = np.percentile(output_dict[v], 50, axis=0)
                    
                    # Calculate metrics for default parameters
                    default_pred = default_output_dict[v][~np.isnan(obs_plot)]
                    default_rmse = np.sqrt(np.mean((default_pred - obs_valid)**2))
                    default_mae = np.mean(np.abs(default_pred - obs_valid))
                    default_r2 = np.corrcoef(default_pred, obs_valid)[0,1]**2 if len(obs_valid) > 1 else 0
                    
                    # Calculate metrics for MCMC median
                    mcmc_pred = median[~np.isnan(obs_plot)]
                    mcmc_rmse = np.sqrt(np.mean((mcmc_pred - obs_valid)**2))
                    mcmc_mae = np.mean(np.abs(mcmc_pred - obs_valid))
                    mcmc_r2 = np.corrcoef(mcmc_pred, obs_valid)[0,1]**2 if len(obs_valid) > 1 else 0
                    
                    # Calculate metrics for 95% CI coverage
                    lower_valid = lower[~np.isnan(obs_plot)]
                    upper_valid = upper[~np.isnan(obs_plot)]
                    coverage = np.mean((obs_valid >= lower_valid) & (obs_valid <= upper_valid)) * 100
                    
                    # Calculate improvement
                    improvement_rmse = (default_rmse - mcmc_rmse) / default_rmse * 100 if default_rmse > 0 else 0
                    improvement_r2 = mcmc_r2 - default_r2
                    
                    # Write to file
                    f.write(f"  Number of valid observations: {len(obs_valid)}\n")
                    f.write(f"  \n")
                    f.write(f"  Default parameters:\n")
                    f.write(f"    RMSE: {default_rmse:.4f}\n")
                    f.write(f"    MAE:  {default_mae:.4f}\n")
                    f.write(f"    R²:   {default_r2:.4f}\n")
                    f.write(f"  \n")
                    f.write(f"  MCMC optimized:\n")
                    f.write(f"    RMSE: {mcmc_rmse:.4f}\n")
                    f.write(f"    MAE:  {mcmc_mae:.4f}\n")
                    f.write(f"    R²:   {mcmc_r2:.4f}\n")
                    f.write(f"    95% CI coverage: {coverage:.1f}%\n")
                    f.write(f"  \n")
                    f.write(f"  Improvements:\n")
                    f.write(f"    RMSE improvement: {improvement_rmse:.1f}%\n")
                    f.write(f"    R² improvement: {improvement_r2:+.4f}\n")
                    
                    # Store for overall statistics
                    site_default_rmse.append(default_rmse)
                    site_mcmc_rmse.append(mcmc_rmse)
                    site_improvements.append(improvement_rmse)
                    
                    all_default_rmse.append(default_rmse)
                    all_mcmc_rmse.append(mcmc_rmse)
                    all_improvements.append(improvement_rmse)
                    
                else:
                    f.write(f"  Cannot calculate statistics (no valid data or default predictions)\n")
            
            # Target summary
            if site_default_rmse:
                f.write(f"\n{target_kind.upper()} {target} SUMMARY:\n")
                f.write(f"  Average RMSE improvement: {np.mean(site_improvements):.1f}%\n")
                f.write(f"  Variables improved: {sum(1 for imp in site_improvements if imp > 0)}/{len(site_improvements)}\n")
        
        # Overall summary across all targets and variables
        if all_default_rmse:
            f.write(f"\n" + "="*60 + "\n")
            f.write("OVERALL SUMMARY (All Targets & Variables)\n")
            f.write("="*60 + "\n")
            f.write(f"Total variables analyzed: {len(all_default_rmse)}\n")
            f.write(f"Average default RMSE: {np.mean(all_default_rmse):.4f}\n")
            f.write(f"Average MCMC RMSE: {np.mean(all_mcmc_rmse):.4f}\n")
            f.write(f"Average RMSE improvement: {np.mean(all_improvements):.1f}%\n")
            f.write(f"Variables improved: {sum(1 for imp in all_improvements if imp > 0)}/{len(all_improvements)}\n")
            f.write(f"Best improvement: {max(all_improvements):.1f}%\n")
            f.write(f"Worst change: {min(all_improvements):.1f}%\n")
            
            # Parameter summary
            f.write(f"\n" + "PARAMETER SUMMARY" + "\n")
            f.write("-" * 40 + "\n")
            f.write("Best-fit parameters:\n")
            for i, (pname, pval) in enumerate(zip(labels_model, best_parms)):
                parm_mean = np.mean(samples[:, i])
                parm_std = np.std(samples[:, i])
                f.write(f"  {pname:25s}: {pval:10.4f} (mean: {parm_mean:8.4f} ± {parm_std:6.4f})\n")
            
            if fit_error and nerr_parms > 0:
                f.write(f"\nError parameters:\n")
                for i, v in enumerate(myvars):
                    err_idx = ensemble_parms.index('sigma_'+v) - n_model_parms
                    err_mean = np.mean(samples[:, n_model_parms + i])
                    err_std = np.std(samples[:, n_model_parms + i])
                    f.write(f"  sigma_{v:20s}: {best_err_parms[err_idx]:10.4f} (mean: {err_mean:8.4f} ± {err_std:6.4f})\n")

    # Also print a brief summary to console
    print(f"\nSummary statistics written to: {summary_file}")
    if all_default_rmse:
        print(f"Overall RMSE improvement: {np.mean(all_improvements):.1f}%")
        print(f"Variables improved: {sum(1 for imp in all_improvements if imp > 0)}/{len(all_improvements)}")

    #Create corner plot for model parameters only
    samples_model = samples[:, :n_model_parms]

    fig = corner.corner(
        samples_model,
        labels=labels_model,
        show_titles=True,
        title_fmt=".2f",
        plot_density=True,
        plot_contours=True,
        title_kwargs={"fontsize": 10}
    )
    # Add R^2 values to off-diagonal plots
    axes = np.array(fig.axes).reshape((n_model_parms, n_model_parms))
    for i in range(n_model_parms):
        for j in range(i):
            x = samples_model[:, j]
            y = samples_model[:, i]
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            ax = axes[i, j]
            ax.annotate(f"$R^2$={r2:.2f}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize=11, color="blue")
    outdir_corner = MCMC_out + '/plots/corner'
    os.makedirs(outdir_corner, exist_ok=True)
    fig.savefig(f"{outdir_corner}/corner_plot.png")
    plt.close(fig)

    # Write best parameters to ELM netCDF parameter file and text file
    out_nc = MCMC_out + '/clm_params_best.nc'
    write_best_params_to_clm(self, best_parms, labels_model, out_nc)
    # Also save best parameters in a simple text file
    out_txt = MCMC_out + '/best_params.txt'
    with open(out_txt, 'w') as f:
        for i, pname in enumerate(labels_model):
            f.write(f"{pname} {best_parms[i]}\n")
    
def write_best_params_to_clm(self, best_parms, labels_model, out_nc_path):
    #TODO:  allow updating FATES parameters
    # Path to template parameter file (first ensemble member)
    template_nc = os.path.join(self.rundir_UQ, 'g00001', 'clm_params_00001.nc')
    # Copy template to output location
    shutil.copy(template_nc, out_nc_path)

    # Open the copied NetCDF file for modification
    with nc.Dataset(out_nc_path, 'r+') as ds:
        for i, pname in enumerate(labels_model):
            pft_idx = self.ensemble_pfts[i]
            # Try to update the variable
            if pname in ds.variables:
                var = ds.variables[pname]
                if pft_idx is not None and var.ndim == 1:
                    # Assume first dimension is PFT
                    var[pft_idx] = best_parms[i]
                elif pft_idx is not None and var.ndim == 2:
                    # Assume second dimension is PFT
                    var[..., pft_idx] = best_parms[i]
                else:
                    var[:] = best_parms[i]
            else:
                print(f"Warning: Parameter {pname} not found in NetCDF file.")
    print(f"Best-fit parameters written to {out_nc_path}")
