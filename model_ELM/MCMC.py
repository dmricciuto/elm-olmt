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


def sample_from_prior(pmin, pmax, nsamples):
    nparms = len(pmin)
    #Uniform priors
    samples = np.random.uniform(low=np.array(pmin), high=np.array(pmax), \
        size=(nsamples,nparms))
    return samples


def log_posterior(parms, sites, myvars, pmin, pmax, obs, obs_err, nparms_ensemble, nerr_parms, run_surrogate):
    # Uniform priors
    prior = 1.0
    for j in range(nparms_ensemble):
        if (parms[j] < pmin[j] or parms[j] > pmax[j]):
            prior = 0.0
    post = prior
    if prior > 0.0:
        parms_model = parms[0:(nparms_ensemble - nerr_parms)]
        for s in sites:
            output = run_surrogate[s](parms_model.reshape(1, -1), myvars)
            for v in myvars:
                myoutput = output[v].flatten()
                myobs    = np.array(obs[s][v]).flatten()
                myerr    = np.array(obs_err[s][v]).flatten()
                # Mask out invalid observations
                mask = (myobs > -9000) & (myerr > 0)
                #mask = mask & (np.arange(len(myobs)) % 12 > 2) & (np.arange(len(myobs)) % 12 < 11)  # mask out winter months
                if (nerr_parms > 0):
                    myerr[mask] = parms[-len(myvars)+myvars.index(v)]
                # Vectorized calculation
                resid = myoutput[mask] - myobs[mask]
                ri = (resid / myerr[mask]) ** 2
                li = -0.5 * np.log(2.0 * np.pi) - np.log(myerr[mask]) - 0.5 * ri
                post += np.sum(li)
    else:
        post = -9999999
    return post

#-------------------------------- MCMC ------------------------------------------------------

def MCMC(self, myvars, nwalkers=32, nsteps=100, fit_error=True):
    nsites = len(self.all_sites)
    sites = self.all_sites
    pmin, pmax, nparms_ensemble = self.ensemble_pmin, self.ensemble_pmax, \
        self.nparms_ensemble
    run_surrogate = {}
    obs = {}
    obs_err = {}
    thiscase={}
    for s in sites:
        if s == sites[0]:
            obs[s] = self.obs.copy()
            obs_err[s] = self.obs_err.copy()
            run_surrogate[s] = self.run_surrogate
        else:
            from model_ELM import ELMcase
            #Get the case objects for other sites
            thiscase[s] = ELMcase(casename=self.casename.replace(self.site, s))
            run_surrogate[s] = thiscase[s].run_surrogate
            obs[s] = thiscase[s].obs.copy()
            obs_err[s] = thiscase[s].obs_err.copy()

    #Add parameters to estimate observation error stddev
    nerr_parms = 0    
    ensemble_parms = self.ensemble_parms.copy()
    if (fit_error):
        print("Fitting observation error parameters")
        for v in myvars:
            #Using the first site to set prior bounds
            mask = (obs[sites[0]][v] > -9000) & (obs_err[sites[0]][v] > 0)
            max_obs = max([np.max(np.abs(obs[sites[0]][v][mask])), 0.01])
            err_prior_min = 0.0
            err_prior_max = 0.25 * max_obs

            # Add error parameter to prior ranges and names
            pmin = np.append(pmin, err_prior_min)
            pmax = np.append(pmax, err_prior_max)
            ensemble_parms = ensemble_parms + ['sigma_'+v]
            nparms_ensemble = len(ensemble_parms)
            nerr_parms = nerr_parms+1

    # Initialize walkers in the prior space
    p0 = sample_from_prior(pmin, pmax, nwalkers)

    # Set up the sampler and run MCMC
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            nparms_ensemble,
            log_posterior,
            args=(sites, myvars, pmin, pmax, obs, obs_err, nparms_ensemble, nerr_parms, run_surrogate),
            #pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    #Get the samples, likelihoods and best parameters
    n_model_parms = len(ensemble_parms) - nerr_parms
    samples = sampler.get_chain(discard=nsteps//5, thin=5, flat=True)
    log_probs = sampler.get_log_prob(discard=nsteps//5, thin=5, flat=True)
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
    outdir = './UQ_output/'+self.casename+'/MCMC_output/plots/pdfs'
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
    for s in sites:
        output_dict = {v: [] for v in myvars}
        for i in range(n_samples):
            parms_model = samples[i, :nparms_ensemble - nerr_parms]
            output = run_surrogate[s](parms_model.reshape(1, -1), myvars)
            for v in myvars:
                output_dict[v].append(output[v].flatten())

        # Convert lists to arrays
        for v in myvars:
            output_dict[v] = np.array(output_dict[v])  # shape: (n_samples, n_obs)

        # Plot predictions with 95% confidence intervals
        outdir_pred = './UQ_output/' + self.casename.replace(self.site, s) + '/MCMC_output/plots/predictions'
        os.makedirs(outdir_pred, exist_ok=True)
        for v in myvars:
            # Compute percentiles
            lower = np.percentile(output_dict[v], 2.5, axis=0)
            upper = np.percentile(output_dict[v], 97.5, axis=0)
            median = np.percentile(output_dict[v], 50, axis=0)
            x = np.arange(len(median))
            obs_plot = np.array(obs[s][v].copy())
            obs_plot[obs_plot < -9000] = np.NaN
            obs_err_plot = np.array(obs_err[s][v].copy())
            obs_err_plot[obs_err_plot < -9000] = np.NaN
            if (fit_error):
                err_idx = ensemble_parms.index('sigma_'+v)
                obs_err_plot[obs_err_plot > -9000] = best_err_parms[err_idx - n_model_parms]

            plt.figure()
            plt.fill_between(x, lower, upper, color='gray', alpha=0.5, label='95% CI')
            plt.plot(x, median, 'r', label='Model median')
            plt.errorbar(x, obs_plot, yerr=obs_err_plot, fmt='o', label='Observations')
            plt.xlabel('Time')
            plt.ylabel(v)
            plt.title(f'Posterior predictive for {v}')
            plt.legend()
            plt.savefig(f'{outdir_pred}/Predictions_{v}_posterior.png')
            plt.close()

    #Create corner plot for model parameters only
    samples_model = samples[:, :n_model_parms]
    labels_model = ensemble_parms[:n_model_parms]

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
    outdir_corner = './UQ_output/' + self.casename + '/MCMC_output/plots/corner'
    os.makedirs(outdir_corner, exist_ok=True)
    fig.savefig(f"{outdir_corner}/corner_plot.png")
    plt.close(fig)

    # Write best parameters to ELM netCDF parameter file and text file
    out_nc = './UQ_output/' + self.casename + '/MCMC_output/clm_params_best.nc'
    write_best_params_to_clm(self, best_parms, labels_model, out_nc)
    # Also save best parameters in a simple text file
    out_txt = './UQ_output/' + self.casename + '/MCMC_output/best_params.txt'
    with open(out_txt, 'w') as f:
        for i, pname in enumerate(labels_model):
            f.write(f"{pname} {best_parms[i]}\n")
    
def write_best_params_to_clm(self, best_parms, labels_model, out_nc_path):
    # Path to template parameter file (first ensemble member)
    template_nc = os.path.join(self.runroot, 'UQ', self.casename, 'g00001', 'clm_params_00001.nc')
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

