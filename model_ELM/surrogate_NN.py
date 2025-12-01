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
    n_processes = min(n_cores - 1, 8)
    print(f"Using up to {n_processes} processes for training")
    
    # Initialize tracking variables
    self.svd_components = {}
    self.use_svd = {}
    
    for var in myvars:
        print(f"Training surrogate for {var}")
        vname = var
        nqoi = self.output[vname].shape[0]  # Number of timesteps

        # Extract outputs and samples 
        y = self.output[vname].transpose()  # Shape: (nsamples, ntimesteps)
        p = self.samples.transpose()        # Shape: (nsamples, nparams)

        # Filter out invalid samples
        valid_indices = np.where(y[:,0].squeeze() > -9999)[0]   

        y = y[valid_indices, :].copy()
        p = p[valid_indices, :].copy()
        
        print(f"Using {len(valid_indices)} valid samples out of {self.samples.shape[1]}")
        # Decide whether to use SVD or per-timestep approach
        if nqoi > 50:
            print(f"  {nqoi} timesteps > 50, using SVD approach")
            self.use_svd[vname] = True
            # SVD will optimize process count internally
            self.train_svd_surrogate(vname, y, p, n_processes)
        else:
            print(f"  {nqoi} timesteps <= 50, using per-timestep approach")
            self.use_svd[vname] = False
            # Optimize for per-timestep too
            n_processes_timestep = min(n_processes, nqoi)
            if n_processes_timestep < n_processes:
                print(f"  Reducing processes from {n_processes} to {n_processes_timestep} (limited by timesteps)")
            self.train_timestep_surrogate(vname, y, p, n_processes_timestep)

def train_svd_surrogate(self, vname, y, p, n_processes):
    """Train surrogate using SVD decomposition - FIXED VERSION"""
    nsamples, nqoi = y.shape
    
    # Perform SVD on the output matrix
    print(f"  Performing SVD on {nsamples}x{nqoi} output matrix")
    
    # Center the data
    y_mean = np.mean(y, axis=0)
    y_centered = y - y_mean
    
    # SVD: y_centered = U @ S @ Vt
    U, S, Vt = np.linalg.svd(y_centered, full_matrices=False)
    
    # Calculate explained variance ratio
    explained_variance_ratio = (S**2) / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components for 99% variance
    n_components = np.argmax(cumulative_variance >= 0.99) + 1
    n_components = max(n_components, 5)
    n_components = min(n_components, min(nsamples-1, nqoi))
    
    print(f"  Using {n_components} SVD components (captures {cumulative_variance[n_components-1]:.1%} variance)")
    
    # OPTIMIZE: Don't use more processes than components
    n_processes_svd = min(n_processes, n_components)
    if n_processes_svd < n_processes:
        print(f"  Reducing processes from {n_processes} to {n_processes_svd} (limited by SVD components)")
    
    # Store SVD information
    self.svd_components[vname] = {
        'mean': y_mean,
        'Vt': Vt[:n_components, :],  # First n_components rows of Vt
        'S': S[:n_components],
        'n_components': n_components,
        'explained_variance_ratio': explained_variance_ratio[:n_components]
    }

    svd_coefficients = U[:, :n_components] * S[:n_components]  # Element-wise multiplication    
    # Alternative correct way:
    # svd_coefficients = y_centered @ Vt[:n_components, :].T
    
    print(f"  SVD coefficient ranges:")
    for c in range(min(5, n_components)):
        coeff = svd_coefficients[:, c]
        print(f"    Component {c}: range=[{np.min(coeff):.2e}, {np.max(coeff):.2e}], std={np.std(coeff):.2e}")
    
    # Now train surrogate models for each SVD coefficient
    print(f"  Training {n_components} SVD coefficient models")
    
    # Shared parameter scaler
    pscaler_shared = preprocessing.StandardScaler().fit(p)
    p_norm = pscaler_shared.transform(p)
    
    # Initialize storage for SVD approach
    self.surrogate[vname] = {}
    self.pscaler[vname] = pscaler_shared
    self.yscaler[vname] = {}
    
    # For SVD, track failed components differently
    self.svd_failed_components = {vname: []}
    self.svd_component_means = {vname: []}
    
    # Prepare training arguments
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50,25), (100,50)],  # Larger networks for SVD
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['adaptive'],
    }
    
    training_args = []
    for c in range(n_components):
        y_c = svd_coefficients[:, c]  # SVD coefficient c
        ptrain, pval, ytrain, yval = train_test_split(
            p_norm, y_c, test_size=0.2, random_state=42
        )
        training_args.append((c, ptrain, ytrain, pval, yval, param_grid, 42))
    
    # Train SVD coefficients in parallel with optimized process count
    print(f"  Training {n_components} SVD coefficients using {n_processes_svd} processes")
    
    with ProcessPoolExecutor(max_workers=n_processes_svd) as executor:
        results = list(executor.map(train_single_timestep, training_args))
    
    # Process results for SVD
    r2_scores = []
    for c, grid, yscaler_c, r2, status in results:
        if grid is not None:
            self.surrogate[vname][c] = grid
            self.yscaler[vname][c] = yscaler_c
            r2_scores.append(r2)
            print(f"    SVD component {c}: R² = {r2:.3f}")
        else:
            print(f"    SVD component {c}: {status} - will use zero coefficient")
            # Store as failed but don't break reconstruction
            self.surrogate[vname][c] = None  
            self.yscaler[vname][c] = None
            self.svd_failed_components[vname].append(c)
            self.svd_component_means[vname].append(np.mean(svd_coefficients[:, c]))
    
    print(f"  {vname} SVD summary: {len(r2_scores)}/{n_components} successful models")
    if r2_scores:
        print(f"  Average R²: {np.mean(r2_scores):.3f}")

def train_timestep_surrogate(self, vname, y, p, n_processes):
    """Train per-timestep surrogates (original approach)"""
    nqoi = y.shape[1]
    
    # Initialize qoi_bad only for per-timestep approach
    if not hasattr(self, 'qoi_bad'):
        self.qoi_bad = {}
        self.qoi_bad_meanval = {}
    
    self.qoi_bad[vname] = []
    self.qoi_bad_meanval[vname] = []
    
    # Shared parameter scaler
    pscaler_shared = preprocessing.StandardScaler().fit(p)
    p_norm = pscaler_shared.transform(p)
    
    # Initialize storage
    self.surrogate[vname] = {}
    self.pscaler[vname] = {}
    self.yscaler[vname] = {}
    
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
            if t % 12 == 0:  # Print every 12th timestep
                print(f"    Timestep {t}: R² = {r2:.3f}")
        else:
            print(f"    Timestep {t}: {status}")
            self.qoi_bad[vname].append(t)
            self.qoi_bad_meanval[vname].append(np.mean(y[:, t]))
    
    print(f"  {vname} summary: {len(timestep_r2_scores)} successful models")
    if timestep_r2_scores:
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
        valid_indices = np.where(y[:,0].squeeze() > -9999)[0]   

        y_true = y[valid_indices, :].copy()
        p_filtered = p[valid_indices, :].copy()
        
        # USE RUN_SURROGATE TO GET PREDICTIONS - HANDLES BOTH SVD AND TIMESTEP
        surrogate_output = self.run_surrogate(p_filtered, [var])
        y_pred = surrogate_output[var]  # Shape: (nsamples, ntimesteps)
        
        nqoi = y_true.shape[1]
        nsamples = y_true.shape[0]
        
        # Calculate R² for each timestep (works for both SVD and per-timestep)
        r2_scores = []
        timestep_indices = []
        
        for t in range(nqoi):
            try:
                # Check if we have valid data for this timestep
                y_true_t = y_true[:, t]
                y_pred_t = y_pred[:, t]
                
                # Remove invalid values
                valid_mask = np.isfinite(y_true_t) & np.isfinite(y_pred_t) & (y_true_t > -9999)
                
                if np.sum(valid_mask) > 5 and np.std(y_true_t[valid_mask]) > 1e-10:  # Need some variation
                    r2 = np.corrcoef(y_true_t[valid_mask], y_pred_t[valid_mask])[0,1]**2
                    if np.isfinite(r2):
                        r2_scores.append(r2)
                        timestep_indices.append(t)
                        
                        # Print periodic updates for SVD to show it's working
                        if self.use_svd[vname] and t % 50 == 0:
                            print(f"  Timestep {t}: R² = {r2:.3f} (from SVD reconstruction)")
                            
            except Exception as e:
                # Skip problematic timesteps
                pass
        
        print(f"  Calculated R² for {len(r2_scores)}/{nqoi} timesteps")
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Added extra subplot for SVD info
        
        # Plot 1: Time series comparison (first few samples)
        ax1 = axes[0, 0]
        time_axis = np.arange(nqoi)
        n_samples_plot = min(5, nsamples)

        # Define colors for different samples
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i in range(n_samples_plot):
            color = colors[i % len(colors)]
            
            # Model (solid line) and surrogate (dashed line) with same color
            ax1.plot(time_axis, y_true[i, :], '-', color=color, alpha=0.7, linewidth=1.5, 
                     label=f'Model {i+1}' if i < 3 else '')
            ax1.plot(time_axis, y_pred[i, :], '--', color=color, alpha=0.7, linewidth=1.5,
                     label=f'Surrogate {i+1}' if i < 3 else '')

        method_text = "SVD" if self.use_svd[vname] else "Per-timestep"
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel(f'{var}')
        ax1.set_title(f'Time Series: Model vs Surrogate ({method_text})')
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
        
        overall_r2 = 0
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
        
        # Plot 3: R² by timestep (FIXED FOR BOTH APPROACHES)
        ax3 = axes[1, 0]
        if r2_scores:
            ax3.plot(timestep_indices, r2_scores, 'bo-', markersize=2, linewidth=1)
            ax3.axhline(y=np.mean(r2_scores), color='r', linestyle='--', 
                       label=f'Mean R² = {np.mean(r2_scores):.3f}')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('R²')
        ax3.set_title(f'R² by Timestep ({method_text})')
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
        
        # Plot 5: Method-specific information
        ax5 = axes[0, 2]
        if self.use_svd[vname]:
            # Plot SVD component importance
            svd_info = self.svd_components[vname]
            n_components = svd_info['n_components']
            
            # Plot explained variance by component
            cumvar = np.cumsum(svd_info['explained_variance_ratio'])
            components = np.arange(1, n_components + 1)
            
            bars = ax5.bar(components, svd_info['explained_variance_ratio'], alpha=0.7, color='skyblue')
            ax5_twin = ax5.twinx()
            ax5_twin.plot(components, cumvar, 'ro-', linewidth=2, markersize=4)
            ax5_twin.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='99% threshold')
            
            ax5.set_xlabel('SVD Component')
            ax5.set_ylabel('Explained Variance Ratio', color='blue')
            ax5_twin.set_ylabel('Cumulative Variance', color='red')
            ax5.set_title(f'SVD Components\n({n_components} components)')
            ax5_twin.legend()
            ax5.grid(True, alpha=0.3)
            
            # Show failed components info
            if hasattr(self, 'svd_failed_components') and vname in self.svd_failed_components:
                n_failed = len(self.svd_failed_components[vname])
                ax5.text(0.05, 0.95, f'Failed Components: {n_failed}/{n_components}', 
                       transform=ax5.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))
            
        else:
            # Show timestep success rate for per-timestep approach
            n_total_timesteps = nqoi
            n_good_timesteps = len(r2_scores)
            n_bad_timesteps = len(self.qoi_bad[vname]) if hasattr(self, 'qoi_bad') and vname in self.qoi_bad else 0
            
            labels = ['Good Models', 'Failed Models', 'Low Variance']
            sizes = [n_good_timesteps, n_bad_timesteps, n_total_timesteps - n_good_timesteps - n_bad_timesteps]
            colors_pie = ['lightgreen', 'red', 'gray']
            
            # Remove zero-size slices
            non_zero = [i for i, size in enumerate(sizes) if size > 0]
            sizes = [sizes[i] for i in non_zero]
            labels = [labels[i] for i in non_zero]
            colors_pie = [colors_pie[i] for i in non_zero]
            
            if sizes:
                ax5.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%')
                ax5.set_title(f'Timestep Model Quality\n({n_total_timesteps} total timesteps)')
        
        # Plot 6: Component coefficients or model distribution (NEW)
        ax6 = axes[1, 2]
        if self.use_svd[vname]:
            # Show first few SVD temporal patterns (Vt rows)
            svd_info = self.svd_components[vname]
            time_axis = np.arange(nqoi)
            n_components_plot = min(4, svd_info['n_components'])
            
            for c in range(n_components_plot):
                ax6.plot(time_axis, svd_info['Vt'][c, :], 
                        label=f'Component {c+1} ({svd_info["explained_variance_ratio"][c]:.1%})')
            
            ax6.set_xlabel('Timestep')
            ax6.set_ylabel('Component Value')
            ax6.set_title('SVD Temporal Patterns')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            # Show R² distribution for per-timestep models
            if r2_scores:
                ax6.hist(r2_scores, bins=20, alpha=0.7, color='skyblue')
                ax6.axvline(x=np.mean(r2_scores), color='r', linestyle='--', 
                           label=f'Mean = {np.mean(r2_scores):.3f}')
                ax6.set_xlabel('R² Score')
                ax6.set_ylabel('Number of Timesteps')
                ax6.set_title('R² Distribution Across Timesteps')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
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
            
            print(f"  {var} Summary ({method_text}):")
            print(f"    Overall R²: {overall_r2:.3f}")
            print(f"    RMSE: {rmse:.3f}")
            print(f"    MAE: {mae:.3f}")
            if r2_scores:
                print(f"    Mean timestep R²: {np.mean(r2_scores):.3f}")
                print(f"    R² range: {np.min(r2_scores):.3f} - {np.max(r2_scores):.3f}")
            print(f"    Valid timesteps: {len(r2_scores)}/{nqoi}")
            if self.use_svd[vname]:
                n_components = self.svd_components[vname]['n_components']
                total_var = self.svd_components[vname]['explained_variance_ratio'].sum()
                print(f"    SVD components: {n_components} (capturing {total_var:.1%} variance)")
            print(f"    Plot saved: surrogate/surrogate_performance_{var}.png")


def run_surrogate(self, parms, myvars):
    """Run surrogate with SVD or per-timestep approach"""
    parms = np.asarray(parms)
    
    if parms.ndim == 1:
        parms = parms.reshape(1, -1)
    elif parms.ndim != 2:
        raise ValueError(f"parms must be 1D or 2D array, got {parms.ndim}D")
    
    nsamples, nparms = parms.shape
    
    if nparms != self.nparms_ensemble:
        raise ValueError(f"Parameter count mismatch: got {nparms}, expected {self.nparms_ensemble}")
    
    surrogate_output = {}
    
    for var in myvars:
        vname = var
        
        if vname not in self.surrogate or not self.surrogate[vname]:
            print(f"Warning: No trained surrogate models found for {vname}")
            surrogate_output[var] = np.zeros((nsamples, self.output[vname].shape[0]))
            continue
        
        if self.use_svd[vname]:
            surrogate_output[var] = self.run_svd_surrogate(parms, vname, nsamples)
        else:
            surrogate_output[var] = self.run_timestep_surrogate(parms, vname, nsamples)
    
    return surrogate_output

def run_svd_surrogate(self, parms, vname, nsamples):
    """Run SVD-based surrogate - FIXED VERSION"""
    # Get SVD components
    svd_info = self.svd_components[vname]
    n_components = svd_info['n_components']
    
    # Normalize parameters
    parms_norm = self.pscaler[vname].transform(parms)
    
    # Predict SVD coefficients
    svd_coefficients_pred = np.zeros((nsamples, n_components))
    
    for c in range(n_components):
        if self.surrogate[vname][c] is None:
            svd_coefficients_pred[:, c] = 0.0
        else:
            try:
                # Predict SVD coefficient c
                y_pred_norm = self.surrogate[vname][c].predict(parms_norm)
                svd_coefficients_pred[:, c] = self.yscaler[vname][c].inverse_transform(
                    y_pred_norm.reshape(-1, 1)
                ).flatten()
            except Exception as e:
                print(f"Warning: SVD coefficient {c} prediction failed: {e}")
                svd_coefficients_pred[:, c] = 0.0
    
    y_pred = svd_info['mean'] + svd_coefficients_pred @ svd_info['Vt']
    
    return y_pred

def run_timestep_surrogate(self, parms, vname, nsamples):
    """Run per-timestep surrogate (original approach)"""
    nqoi = self.output[vname].shape[0]
    y_pred = np.zeros((nsamples, nqoi))
    
    # Check if qoi_bad exists for this variable (per-timestep only)
    if not hasattr(self, 'qoi_bad') or vname not in self.qoi_bad:
        print(f"Warning: No qoi_bad info for {vname} - assuming all timesteps valid")
        bad_timesteps = []
    else:
        bad_timesteps = self.qoi_bad[vname]
    
    # Get first good timestep for parameter scaling
    try:
        first_good_timestep = next(t for t in range(nqoi) if t not in bad_timesteps)
        parms_norm = self.pscaler[vname][first_good_timestep].transform(parms)
    except StopIteration:
        print(f"Warning: No valid timesteps found for {vname}")
        return y_pred
    
    # Predict each timestep
    for t in range(nqoi):
        if t in bad_timesteps:
            idx = bad_timesteps.index(t)
            y_pred[:, t] = self.qoi_bad_meanval[vname][idx]
        else:
            try:
                y_pred_norm = self.surrogate[vname][t].predict(parms_norm)
                y_pred[:, t] = self.yscaler[vname][t].inverse_transform(
                    y_pred_norm.reshape(-1, 1)
                ).flatten()
            except Exception as e:
                print(f"Warning: Timestep {t} prediction failed: {e}")
                y_pred[:, t] = 0.0
    
    return y_pred


