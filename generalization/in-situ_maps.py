import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from dtw import dtw

from config_fit import DATA_MANIFEST, DEVICE
from hux import HUXPropagator
from model import WSASurrogateModel
from dataset_fit import CarringtonDataset
import pandas as pd

# Load best model checkpoint
model = WSASurrogateModel(img_size=(360, 180)).to(DEVICE)
ckpt = torch.load(os.path.join('checkpoints', 'last.pt'), map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Instantiate 1D HUX propagator
hux1d = HUXPropagator().to(DEVICE)

# Inline 2D HUX for polar plots
def hux_full(v_in, omega, r_start=0.1, r_end=1.0, n_rad=512):
    """
    Propagate a 1D boundary v_in (shape [phi]) from r_start→r_end over n_rad steps.
    Returns a 2D array [n_rad, phi].
    """
    phi_count = v_in.shape[0]
    dr_AU = (r_end - r_start) / (n_rad - 1)
    dr_km = dr_AU * 1.496e8
    dphi = 2 * np.pi / phi_count

    v_hux = np.zeros((n_rad, phi_count), dtype=float)
    v_hux[0, :] = v_in.astype(float)
    for i in range(1, n_rad):
        prev = v_hux[i-1]
        rolled = np.roll(prev, -1)
        v_hux[i] = prev + (dr_km * omega / dphi) * ((rolled - prev) / prev)
    return v_hux

def apply_wsa_equation(exp_factor, mad, params):
    """Apply WSA equation with given parameters"""
    vmin = params['vmin']
    vmax = params['vmax']
    alpha = params['alpha']
    beta = params['beta']
    a1 = params['a1']
    a2 = params['a2']
    w = params['w']
    
    # Denormalize channel values if needed
    if exp_factor.max() <= 1.0:  # Check if normalized
        exp_factor_denorm = exp_factor * 10.0 - 4.0     # (max-min)=10, min=-4
        mad_denorm = mad * 25.0                         # (max-min)=25, min=0
    else:
        exp_factor_denorm = exp_factor
        mad_denorm = mad
        
    mad_denorm = mad_denorm * (np.pi / 180.0)    # Convert degrees to radians
    
    # WSA equation calculation
    power_term = 1 + np.power(10, exp_factor_denorm)
    term1 = vmax / np.power(power_term, alpha)
    
    exp_term = np.exp(-1 * np.power((mad_denorm / w), beta))
    term2 = np.power(1 - a1 * exp_term, a2)
    
    wsa_speed = vmin + term1 * term2
    
    return wsa_speed

def plot_cr_analysis(target_cr=2053):
    """
    Create a comprehensive plot for a specific CR with modified layout:
    Row 1: Three 2D equatorial plane slices (WSA Default, WSA Fitted, and WSA+)
    Row 2: 1D speed profile at 0.1 AU
    Row 3: 1D speed profile at 1 AU
    """
    
    # Find the entry for target CR
    target_entry = None
    for entry in DATA_MANIFEST:
        ds = CarringtonDataset([entry])
        sample = ds[0]
        if sample['cr'] == target_cr:
            target_entry = entry
            break
    
    if target_entry is None:
        print(f"CR {target_cr} not found in DATA_MANIFEST")
        return
    
    # Prepare dataset
    ds = CarringtonDataset([target_entry])
    sample = ds[0]
    cr = sample['cr']
    
    print(f"Processing CR {cr}...")
    
    # NN-predicted map at 0.1 AU
    with torch.no_grad():
        x_map = sample['map'].unsqueeze(0).to(DEVICE)
        pred_map = model(x_map)[0,0].cpu().numpy()
    
    # Get Earth latitude indices
    lat_idx = sample['lat_idx'].cpu().numpy()
    T = lat_idx.size
    omega = target_entry['omega']
    
    # Load WSA parameters (if available)
    RESULTS_DIR = "../optimization/results"
    df_params = pd.read_csv('../optimization/results/fitted_wsa_params.csv')
    
    # Load fit scores
    df_fit_scores = pd.read_csv('CR_metrics/fit_scores.csv')
    
    # Default WSA parameters
    wsa_default_params = {
        'vmin': 250.0, 'vmax': 750.0, 'alpha': 0.222, 'beta': 1.25,
        'a1': 0.80, 'w': 0.028, 'a2': 3.0
    }
    
    # Get fitted parameters for this CR
    cr_params = df_params[df_params['CR'] == cr]
    if len(cr_params) == 0:
        print(f"No fitted parameters found for CR {cr}, using defaults")
        fitted_params = wsa_default_params.copy()
    else:
        fitted_params = {
            'vmin': cr_params['vmin'].values[0], 'vmax': cr_params['vmax'].values[0],
            'alpha': cr_params['alpha'].values[0], 'beta': cr_params['beta'].values[0],
            'a1': cr_params['a1'].values[0], 'w': cr_params['w'].values[0],
            'a2': cr_params['a2'].values[0]
        }
    
    # Generate WSA maps
    exp_factor = sample['map'][0].cpu().numpy()
    mad = sample['map'][1].cpu().numpy()
    wsa_default_map = apply_wsa_equation(exp_factor, mad, wsa_default_params)
    wsa_fitted_map = apply_wsa_equation(exp_factor, mad, fitted_params)
    
    # Extract boundary values at sub-Earth latitude
    wsa_default_bdy = np.zeros(wsa_default_map.shape[0])
    wsa_fitted_bdy = np.zeros(wsa_fitted_map.shape[0])
    nn_bdy = np.zeros(pred_map.shape[0])
    
    for i in range(wsa_default_map.shape[0]):
        idx = lat_idx[i % lat_idx.size]
        wsa_default_bdy[i] = wsa_default_map[i, idx]
        wsa_fitted_bdy[i] = wsa_fitted_map[i, idx]
        nn_bdy[i] = pred_map[i, idx]
    
    # 2D HUX propagation
    v2d_wsa_default = hux_full(wsa_default_bdy, omega)
    v2d_wsa_fitted = hux_full(wsa_fitted_bdy, omega)
    v2d_nn = hux_full(nn_bdy, omega)
    
    # 1D time series at 0.1 AU
    wsa_default_ts = wsa_default_map[np.arange(T), lat_idx]
    wsa_fitted_ts = wsa_fitted_map[np.arange(T), lat_idx]
    nn_ts = pred_map[np.arange(T), lat_idx]
    
    # 1D HUX propagation to 1 AU
    omega_t = torch.tensor([omega], dtype=torch.float32, device=DEVICE)
    
    v_in_default = torch.from_numpy(wsa_default_ts.astype(np.float32)).unsqueeze(0).to(DEVICE)
    v_in_fitted = torch.from_numpy(wsa_fitted_ts.astype(np.float32)).unsqueeze(0).to(DEVICE)
    v_in_nn = torch.from_numpy(nn_ts.astype(np.float32)).unsqueeze(0).to(DEVICE)
    
    v_out_default = hux1d(v_in_default, omega_t)[0].cpu().numpy()
    v_out_fitted = hux1d(v_in_fitted, omega_t)[0].cpu().numpy()
    v_out_nn = hux1d(v_in_nn, omega_t)[0].cpu().numpy()
    
    # OMNI data
    omni = sample['omni'].cpu().numpy()
    
    # Calculate metrics
    def calc_metrics(pred, obs):
        mae = np.mean(np.abs(pred - obs))
        rmse = np.sqrt(np.mean((pred - obs)**2))
        cc, _ = pearsonr(pred, obs)
        dtw_dist = dtw(pred, obs).normalizedDistance
        return mae, rmse, cc, dtw_dist
    
    mae_def, rmse_def, cc_def, dtw_def = calc_metrics(v_out_default, omni)
    mae_fit, rmse_fit, cc_fit, dtw_fit = calc_metrics(v_out_fitted, omni)
    mae_nn, rmse_nn, cc_nn, dtw_nn = calc_metrics(v_out_nn, omni)
    
    # Get fit score for this CR
    fit_score_row = df_fit_scores[df_fit_scores['cr'] == cr]
    if len(fit_score_row) > 0:
        fit_score = fit_score_row['fit_score'].values[0]
    else:
        fit_score = np.nan  # If not found
    
    # Create the plot with modified layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, height_ratios=[1, 0.8, 0.8], figure=fig)
    
    # Prepare polar coordinates for 2D plots
    phi = np.linspace(0, 2*np.pi, v2d_wsa_fitted.shape[1], endpoint=False)
    R, Phi = np.meshgrid(np.linspace(0.1, 1.0, v2d_wsa_fitted.shape[0]), phi, indexing='ij')
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)
    
    cmap = 'viridis'
    # Row 1, Column 1: WSA Default + HUX 2D
    ax1 = fig.add_subplot(gs[0, 0])
    pcm1 = ax1.pcolormesh(X, Y, v2d_wsa_default, shading='auto', vmin=200, vmax=700, cmap=cmap)
    ax1.set_aspect('equal')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_title(f'a) CR{cr} WSA Default', fontsize=14)
    ax1.set_xlabel('X (AU)', fontsize=14)
    ax1.set_ylabel('Y (AU)', fontsize=14)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', length=6, labelsize=11)
    ax1.tick_params(axis='both', which='minor', length=3, labelsize=11)
    cbar1 = plt.colorbar(pcm1, ax=ax1, shrink=0.8)

    
    # Row 1, Column 2: WSA Fitted + HUX 2D
    ax2 = fig.add_subplot(gs[0, 1])
    pcm2 = ax2.pcolormesh(X, Y, v2d_wsa_fitted, shading='auto', vmin=200, vmax=700, cmap=cmap)
    ax2.set_aspect('equal')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title(f'b) CR{cr} WSA Optimized', fontsize=14)
    ax2.set_xlabel('X (AU)', fontsize=14)
    ax2.set_ylabel('Y (AU)', fontsize=14)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='major', length=6, labelsize=11)
    ax2.tick_params(axis='both', which='minor', length=3, labelsize=11)
    cbar2 = plt.colorbar(pcm2, ax=ax2, shrink=0.8)
    
    # Row 1, Column 3: NN + HUX 2D
    ax3 = fig.add_subplot(gs[0, 2])
    pcm3 = ax3.pcolormesh(X, Y, v2d_nn, shading='auto', vmin=200, vmax=700, cmap=cmap)
    ax3.set_aspect('equal')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_title(f'c) CR{cr} WSA+', fontsize=14)
    ax3.set_xlabel('X (AU)', fontsize=14)
    ax3.set_ylabel('Y (AU)', fontsize=14)
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax3.minorticks_on()
    ax3.tick_params(axis='both', which='major', length=6, labelsize=11)
    ax3.tick_params(axis='both', which='minor', length=3, labelsize=11)
    cbar3 = plt.colorbar(pcm3, ax=ax3, shrink=0.8)
    
    # Row 2: 1D speed profile at 0.1 AU
    ax4 = fig.add_subplot(gs[1, :])
    t = np.arange(T)
    ax4.plot(t, wsa_default_ts, label='WSA Default', color='C0', linestyle='-', linewidth=2)
    ax4.plot(t, wsa_fitted_ts, label='WSA Optimized', color='C2', linestyle='--', linewidth=2)
    ax4.plot(t, nn_ts, label='WSA+', color='C1', linewidth=2)
    ax4.set_ylim(250, 870)
    ax4.set_title('d) Speed Profile at 0.1 AU (Sub-Earth Latitude)', fontsize=16)
    ax4.set_xlabel('Time Index', fontsize=14)
    ax4.set_ylabel('Speed (km/s)', fontsize=14, labelpad=15)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax4.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax4.minorticks_on()
    ax4.tick_params(axis='both', which='major', length=6, labelsize=12)
    ax4.tick_params(axis='both', which='minor', length=3, labelsize=12)
    ax4.legend(ncol=3, loc='upper center', fontsize=14)
    
    # Row 3: 1D speed profile at 1 AU with metrics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(t, v_out_default, 
             label=f'WSA Default  (${{CC={cc_def:.2f}, RMSE={rmse_def:.1f}, MAE={mae_def:.1f}, DTW={dtw_def:.2f}}}$)', 
             color='C0', linestyle='-', linewidth=2)
    ax5.plot(t, v_out_fitted, 
             label=f'WSA Optimized  (${{CC={cc_fit:.2f}, RMSE={rmse_fit:.1f}, MAE={mae_fit:.1f}, DTW={dtw_fit:.2f}}}$)', 
             color='C2', linestyle='--', linewidth=2)
    if not np.isnan(fit_score):
        ax5.plot(t, v_out_nn, 
                 label=f'WSA+  (${{CC={cc_nn:.2f}, RMSE={rmse_nn:.1f}, MAE={mae_nn:.1f}, DTW={dtw_nn:.2f}}}$)', 
                 color='C1', linewidth=2)
    else:
        ax5.plot(t, v_out_nn, 
                 label=f'WSA+  (${{CC={cc_nn:.2f}, RMSE={rmse_nn:.1f}, MAE={mae_nn:.1f}, DTW={dtw_nn:.2f}}}$)', 
                 color='C1', linewidth=2)
        
    ax5.plot(t, omni, label='OMNI', color='black', alpha=0.7, linewidth=2)
    ax5.set_ylim(250, 890)
    ax5.set_title('e) Speed Profile at 1 AU vs OMNI Data', fontsize=16)
    ax5.set_xlabel('Time Index', fontsize=14)
    ax5.set_ylabel('Speed (km/s)', fontsize=14, labelpad=15)
    ax5.tick_params(axis='both', which='major', labelsize=12)
    ax5.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax5.minorticks_on()
    ax5.tick_params(axis='both', which='major', length=6, labelsize=12)
    ax5.tick_params(axis='both', which='minor', length=3, labelsize=12)
    ax5.legend(ncol=2, loc='upper center', fontsize=13)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('CR_metrics', exist_ok=True)
    save_path = f'CR_metrics/CR_{cr}_in-situ_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {save_path}")
    print(f"Metrics for CR {cr}:")
    print(f"  WSA Default: CC={cc_def:.3f}, RMSE={rmse_def:.1f}, MAE={mae_def:.1f}")
    print(f"  WSA Fitted:  CC={cc_fit:.3f}, RMSE={rmse_fit:.1f}, MAE={mae_fit:.1f}")
    print(f"  WSA+:        CC={cc_nn:.3f}, RMSE={rmse_nn:.1f}, MAE={mae_nn:.1f}")

if __name__ == "__main__":
    print("Creating modified CR analysis plot...")
    plot_cr_analysis(target_cr=2065)
    print("Done!")