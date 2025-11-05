from netCDF4 import Dataset
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, math, sys
import numpy as np
import pickle
from optparse import OptionParser
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Suppress sklearn warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_single_timestep(args):
    """Train a single timestep model - designed for multiprocessing"""
    t, ptrain, ytrain, pval, yval, param_grid, random_state = args
    
    # Skip problematic timesteps
    unique_vals = np.unique(ytrain)
    zero_fraction = np.sum(ytrain == 0) / len(ytrain)
    
    if len(unique_vals) <= 2 or zero_fraction > 0.8:
        return t, None, None, None, f"Skipped (constant values or >80% zeros)"
    
    try:
        # Scale outputs for this timestep
        yscaler_t = preprocessing.StandardScaler()
        ytrain_norm = yscaler_t.fit_transform(ytrain.reshape(-1, 1)).flatten()
        
        clf = MLPRegressor(
            max_iter=500,
            early_stopping=True, 
            validation_fraction=0.2,
            n_iter_no_change=10, 
            random_state=random_state
        )
        
        grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=1)  # Single job per timestep
        grid.fit(ptrain, ytrain_norm)
        
        # Calculate R²
        yval_pred = yscaler_t.inverse_transform(
            grid.predict(pval).reshape(-1, 1)
        ).flatten()
        r2 = np.corrcoef(yval, yval_pred)[0,1]**2
        
        return t, grid, yscaler_t, r2, "Success"
        
    except Exception as e:
        return t, None, None, None, f"Training failed: {e}"

def train_surrogate(self, myvars):
    # Determine number of processes
    n_cores = mp.cpu_count()
    n_processes = min(n_cores - 1, 8)  # Leave 1 core free, cap at 8
    print(f"Using {n_processes} processes for timestep-parallel training")
    
    self.qoi_bad = {}
    self.qoi_bad_meanval = {}
    
    for var in myvars:
        print(f"Training per-timestep surrogates for {var}")
        vname = var
        nqoi = self.output[vname].shape[0]  # Number of timesteps

        # Extract outputs and samples 
        y = self.output[vname].transpose()  # Shape: (nsamples, ntimesteps)
        p = self.samples.transpose()        # Shape: (nsamples, nparams)

        # Filter out invalid samples (same as before)
        if ('NPP_correct' in var):
            valid_indices = np.where(np.min(y[:,1:], axis=1) > 400)[0]
        elif ('NPP_response' in var):
            valid_indices = np.where(np.min(y[:,1:], axis=1) > 10)[0]
        elif ('NPP' in var):
            valid_indices = np.where(np.min(y[:,1:], axis=1) > 0)[0]
        else:
            valid_indices = np.where(y[:,0].squeeze() > -9999)[0]   

        y = y[valid_indices, :].copy()
        p = p[valid_indices, :].copy()
        
        print(f"Using {len(valid_indices)} valid samples out of {self.samples.shape[1]}")

        # Shared parameter scaler for all timesteps
        pscaler_shared = preprocessing.StandardScaler().fit(p)
        p_norm = pscaler_shared.transform(p)
        
        # Initialize storage
        self.surrogate[vname] = {}
        self.pscaler[vname] = {}
        self.yscaler[vname] = {}
        self.qoi_bad[vname] = []
        self.qoi_bad_meanval[vname] = []
        
        # Prepare arguments for parallel training
        param_grid = {
            'hidden_layer_sizes': [(20,), (50,), (20,10)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate': ['adaptive'],
        }
        
        training_args = []
        for t in range(nqoi):
            y_t = y[:, t]
            ptrain, pval, ytrain, yval = train_test_split(
                p_norm, y_t, test_size=0.2, random_state=42
            )
            training_args.append((t, ptrain, ytrain, pval, yval, param_grid, 42))
        
        # Train timesteps in parallel
        print(f"  Training {nqoi} timesteps in parallel using {n_processes} processes...")
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(train_single_timestep, training_args))
        
        # Process results
        timestep_r2_scores = []
        for t, grid, yscaler_t, r2, status in results:
            if grid is not None:
                self.surrogate[vname][t] = grid
                self.pscaler[vname][t] = pscaler_shared
                self.yscaler[vname][t] = yscaler_t
                timestep_r2_scores.append(r2)
                print(f"  Timestep {t}: R² = {r2:.3f}")
            else:
                print(f"  Timestep {t}: {status}")
                self.qoi_bad[vname].append(t)
                self.qoi_bad_meanval[vname].append(np.mean(y[:, t]))
        
        print(f"  {var} summary: {len(timestep_r2_scores)} successful models")
        print(f"  Average R²: {np.mean(timestep_r2_scores):.3f}")
        print(f"  Failed timesteps: {len(self.qoi_bad[vname])}")


def plot_surrogate(self, myvars):
    """Plot surrogate model performance for each variable"""
    
    for var in myvars:
        vname = var
        
        if vname not in self.surrogate or not self.surrogate[vname]:
            print(f"No surrogate models to plot for {vname}")
            continue
            
        print(f"Plotting surrogate performance for {var}")
        
        # Get the original training data
        y = self.output[vname].transpose()  # Shape: (nsamples, ntimesteps)
        p = self.samples.transpose()        # Shape: (nsamples, nparams)
        
        # Apply same filtering as in training
        if ('NPP_correct' in var):
            valid_indices = np.where(np.min(y[:,1:], axis=1) > 400)[0]
        elif ('NPP_response' in var):
            valid_indices = np.where(np.min(y[:,1:], axis=1) > 10)[0]
        elif ('NPP' in var):
            valid_indices = np.where(np.min(y[:,1:], axis=1) > 0)[0]
        else:
            valid_indices = np.where(y[:,0].squeeze() > -9999)[0]   

        y_true = y[valid_indices, :].copy()
        p_filtered = p[valid_indices, :].copy()
        
        surrogate_output = self.run_surrogate(p_filtered, [var])
        y_pred = surrogate_output[var]  # Shape: (nsamples, ntimesteps)
        
        nqoi = y_true.shape[1]
        nsamples = y_true.shape[0]
        
        # Calculate R² for each timestep
        r2_scores = []
        for t in range(nqoi):
            if t not in self.qoi_bad[vname]:
                try:
                    r2 = np.corrcoef(y_true[:, t], y_pred[:, t])[0,1]**2
                    r2_scores.append(r2)
                except:
                    pass  # Skip if calculation fails
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Time series comparison (first few samples)
        ax1 = axes[0, 0]
        time_axis = np.arange(nqoi)
        n_samples_plot = min(5, nsamples)

        # Define colors for different samples
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i in range(n_samples_plot):
            color = colors[i % len(colors)]  # Cycle through colors if more than 5 samples
            
            # Model (solid line) and surrogate (dashed line) with same color
            ax1.plot(time_axis, y_true[i, :], '-', color=color, alpha=0.7, linewidth=1.5, 
                     label=f'Model {i+1}' if i < 3 else '')  # Only label first 3 to avoid clutter
            ax1.plot(time_axis, y_pred[i, :], '--', color=color, alpha=0.7, linewidth=1.5,
                     label=f'Surrogate {i+1}' if i < 3 else '')

        ax1.set_xlabel('Timestep')
        ax1.set_ylabel(f'{var}')
        ax1.set_title(f'Time Series: Model (solid) vs Surrogate (dashed)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: 1:1 scatter plot (all data)
        ax2 = axes[0, 1]
        y_flat = y_true.flatten()
        pred_flat = y_pred.flatten()
        
        # Remove bad values for plotting
        valid_mask = (y_flat > -9999) & (pred_flat > -9999) & np.isfinite(y_flat) & np.isfinite(pred_flat)
        y_valid = y_flat[valid_mask]
        pred_valid = pred_flat[valid_mask]
        
        if len(y_valid) > 0:
            ax2.scatter(y_valid, pred_valid, alpha=0.5, s=1)
            
            # Add 1:1 line
            min_val = min(np.min(y_valid), np.min(pred_valid))
            max_val = max(np.max(y_valid), np.max(pred_valid))
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
            
            # Calculate overall R²
            overall_r2 = np.corrcoef(y_valid, pred_valid)[0,1]**2
            ax2.text(0.05, 0.95, f'R² = {overall_r2:.3f}', transform=ax2.transAxes, 
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel(f'True {var}')
        ax2.set_ylabel(f'Surrogate {var}')
        ax2.set_title('1:1 Plot (All Data)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² by timestep
        ax3 = axes[1, 0]
        if r2_scores:
            good_timesteps = [t for t in range(nqoi) if t not in self.qoi_bad[vname]]
            ax3.plot(good_timesteps, r2_scores, 'bo-', markersize=3)
            ax3.axhline(y=np.mean(r2_scores), color='r', linestyle='--', 
                       label=f'Mean R² = {np.mean(r2_scores):.3f}')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('R²')
        ax3.set_title('R² Score by Timestep')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Plot 4: Residuals histogram
        ax4 = axes[1, 1]
        if len(y_valid) > 0:
            residuals = pred_valid - y_valid
            ax4.hist(residuals, bins=50, alpha=0.7, density=True)
            ax4.axvline(x=0, color='r', linestyle='--', alpha=0.8, label='Zero')
            ax4.axvline(x=np.mean(residuals), color='g', linestyle='--', 
                       label=f'Mean = {np.mean(residuals):.3f}')
        ax4.set_xlabel(f'Residuals (Surrogate - True)')
        ax4.set_ylabel('Density')
        ax4.set_title('Residuals Distribution')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(self.UQ_output+'/surrogate', exist_ok=True)
        plt.savefig(f'{self.UQ_output}/surrogate/surrogate_performance_{var}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        if len(y_valid) > 0:
            rmse = np.sqrt(np.mean((pred_valid - y_valid)**2))
            mae = np.mean(np.abs(pred_valid - y_valid))
            
            print(f"  {var} Summary:")
            print(f"    Overall R²: {overall_r2:.3f}")
            print(f"    RMSE: {rmse:.3f}")
            print(f"    MAE: {mae:.3f}")
            if r2_scores:
                print(f"    Mean timestep R²: {np.mean(r2_scores):.3f}")
            print(f"    Good timesteps: {len(r2_scores)}/{nqoi}")
            print(f"    Plot saved: surrogate/surrogate_performance_{var}.png")


def run_surrogate(self, parms, myvars):
    #Ensure parms is a proper numpy array with correct shape
    parms = np.asarray(parms)
    
    # Handle different input shapes
    if parms.ndim == 1:
        # Single parameter set - reshape to (1, nparms)
        parms = parms.reshape(1, -1)
    elif parms.ndim != 2:
        raise ValueError(f"parms must be 1D or 2D array, got {parms.ndim}D")
    nsamples, nparms = parms.shape
    
    # Validate parameter count matches training data
    expected_nparms = self.nparms_ensemble
    if nparms != expected_nparms:
        raise ValueError(f"Parameter count mismatch: got {nparms}, expected {expected_nparms}")
    
    surrogate_output = {}
    for var in myvars:
        vname = var
        nqoi = self.output[vname].shape[0]
        
        # Initialize output array
        y_pred = np.zeros((nsamples, nqoi))
        
        # Check if we have any trained models for this variable
        if vname not in self.surrogate or not self.surrogate[vname]:
            print(f"Warning: No trained surrogate models found for {vname}")
            surrogate_output[var] = y_pred  # Return zeros
            continue
        
        # Normalize parameters (using first good timestep's scaler since they're all the same)
        try:
            first_good_timestep = next(t for t in range(nqoi) if t not in self.qoi_bad[vname])
            parms_norm = self.pscaler[vname][first_good_timestep].transform(parms)
        except StopIteration:
            print(f"Warning: No valid timesteps found for {vname}")
            surrogate_output[var] = y_pred  # Return zeros
            continue
        
        # Predict each timestep
        for t in range(nqoi):
            if t in self.qoi_bad[vname]:
                # Use mean value for failed timesteps
                idx = self.qoi_bad[vname].index(t)
                y_pred[:, t] = self.qoi_bad_meanval[vname][idx]
            else:
                try:
                    # Predict using timestep-specific model
                    y_pred_norm = self.surrogate[vname][t].predict(parms_norm)
                    y_pred[:, t] = self.yscaler[vname][t].inverse_transform(
                        y_pred_norm.reshape(-1, 1)
                    ).flatten()
                except Exception as e:
                    print(f"Warning: Prediction failed for {vname} timestep {t}: {e}")
                    # Use mean value as fallback
                    y_pred[:, t] = np.mean(self.output[vname][t, :]) if hasattr(self, 'output') else 0.0
        surrogate_output[var] = y_pred
    
    return surrogate_output


