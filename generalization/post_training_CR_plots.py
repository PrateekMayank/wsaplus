import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from dtw import dtw

from config_fit import DATA_MANIFEST, DEVICE
from hux import HUXPropagator
from model import WSASurrogateModel
from dataset_fit import CarringtonDataset
import csv
import pandas as pd

# Output directory for summary plots
PLOTS_DIR = 'CR_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

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

# ------------------------------------------------------------
# Function to apply WSA equation
def apply_wsa_equation(exp_factor, mad, params):
    """Apply WSA equation with given parameters"""
    # Extract parameters
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
# ------------------------------------------------------------

RESULTS_DIR = "../optimization/results"
WSA_MAPS_DIR = os.path.join(RESULTS_DIR, "wsa_maps")

fair_crs_file = os.path.join(RESULTS_DIR, "fair_crs.csv")
if os.path.exists(fair_crs_file):
    fair_crs = pd.read_csv(fair_crs_file)['CR'].tolist()
    print(f"Loaded {len(fair_crs)} fair  CRs from {fair_crs_file}")
else:
    # Fallback to all CRs except excluded ones if file doesn't exist
    print(f"Warning: {fair_crs_file} not found, using all available CRs instead")
    fair_crs = None

# ------------------------------------------------------------
# ------------------------------------------------------------

# Load the optimized params of WSA
df_params = pd.read_csv('../optimization/results/fitted_wsa_params.csv')

# List of parameters
parameters = ['cr', 'vmin', 'vmax', 'alpha', 'beta', 'a1', 'w', 'a2']

# Default WSA parameters
wsa_default_params = {
    'vmin': 250.0,
    'vmax': 750.0,
    'alpha': 0.222,
    'beta': 1.25,
    'a1': 0.80,
    'w': 0.028,
    'a2': 3.0
}

# ------------------------------------------------------------
# ------------------------------------------------------------
# Loop over all Carrington rotations
for i, entry in enumerate(DATA_MANIFEST):

    # Validation Set After Training
    #if i % 10 != 0:
    #    continue

    # Load fair/bad CRs
    #cr = entry['cr']  # Get CR from the entry directly
    #if fair_crs is not None and entry['cr'] not in fair_crs:
    #    print(f"Skipping CR {entry['cr']} as it is not in the fair set")
    #    continue

    # Load NPZ for this CR
    #data_np = np.load(entry['map_path'], allow_pickle=True)
    #cr = int(data_np['cr_info'].item()['cr'])

    # Prepare a single-sample dataset to get necessary indices & OMNI
    ds = CarringtonDataset([entry])
    sample = ds[0]
    cr = sample['cr']

    # NN-predicted map at 0.1 AU
    with torch.no_grad():
        x_map = sample['map'].unsqueeze(0).to(DEVICE)
        pred_map = model(x_map)[0,0].cpu().numpy()
        wsa_map = sample['wsa'].cpu().numpy()

    # Get Earth latitude indices first
    lat_idx = sample['lat_idx'].cpu().numpy()
    T = lat_idx.size

    # 2D polar propagation (WSA & NN)
    omega = entry['omega']

    # Fetch the speed at Sub-Earth latitudes at 0.1 AU
    wsa_bdy_full = np.zeros(wsa_map.shape[0])
    nn_bdy_full = np.zeros(pred_map.shape[0])
    for i in range(wsa_map.shape[0]):
        idx = lat_idx[i % lat_idx.size]  # Handle length mismatch using modulo
        wsa_bdy_full[i] = wsa_map[i, idx]
        nn_bdy_full[i] = pred_map[i, idx]

    # 2D HUX propagation for WSA and NN maps
    v2d_wsa  = hux_full(wsa_bdy_full,  omega)
    v2d_pred = hux_full(nn_bdy_full,   omega)

    # 1D Earth-latitude series at 0.1 AU for NN
    pred_ts = pred_map[np.arange(T), lat_idx]

    # 1D HUX propagate for final 1AU series for NN
    v_in_pred = torch.from_numpy(pred_ts.astype(np.float32)).unsqueeze(0).to(DEVICE)
    omega_t   = torch.tensor([omega], dtype=torch.float32, device=DEVICE)
    v_out_pred = hux1d(v_in_pred, omega_t)[0].cpu().numpy()

    # OMNI series
    omni = sample['omni'].cpu().numpy()

    #---------------------------------------------
    # Prepare fitted & default WSA maps

    # Extract expansion factor and mad from the input maps
    exp_factor = sample['map'][0].cpu().numpy()  # First channel is expansion factor
    mad = sample['map'][1].cpu().numpy()  # Second channel is minimum angular distance

    # Get fitted parameters for this CR from the CSV
    cr_params = df_params[df_params['CR'] == cr]
    if len(cr_params) == 0:
        print(f"No fitted parameters found for CR {cr}, using defaults")
        fitted_params = wsa_default_params.copy()
    else:
        fitted_params = {
            'vmin': cr_params['vmin'].values[0],
            'vmax': cr_params['vmax'].values[0],
            'alpha': cr_params['alpha'].values[0],
            'beta': cr_params['beta'].values[0],
            'a1': cr_params['a1'].values[0],
            'w': cr_params['w'].values[0],
            'a2': cr_params['a2'].values[0]
        }

    # Apply WSA equations to generate maps
    wsa_default_map = apply_wsa_equation(exp_factor, mad, wsa_default_params)
    wsa_fitted_map = apply_wsa_equation(exp_factor, mad, fitted_params)
    
    # Get Earth latitude boundary values
    wsa_default_ts = wsa_default_map[np.arange(T), lat_idx]
    wsa_fitted_ts = wsa_fitted_map[np.arange(T), lat_idx]
    
    # Propagate to 1AU using HUX
    v_in_default = torch.from_numpy(wsa_default_ts.astype(np.float32)).unsqueeze(0).to(DEVICE)
    v_in_fitted = torch.from_numpy(wsa_fitted_ts.astype(np.float32)).unsqueeze(0).to(DEVICE)
    v_out_default = hux1d(v_in_default, omega_t)[0].cpu().numpy()
    v_out_fitted = hux1d(v_in_fitted, omega_t)[0].cpu().numpy()

    #----------------------------------------------

    # Compute metrics at 1AU
    mae_wsa_def = np.mean(np.abs(v_out_default - omni))
    mae_wsa_fit = np.mean(np.abs(v_out_fitted - omni))
    mae_pred = np.mean(np.abs(v_out_pred - omni))
    cc_wsa_def, _  = pearsonr(v_out_default, omni)
    cc_wsa_fit, _  = pearsonr(v_out_fitted, omni)
    cc_pred, _ = pearsonr(v_out_pred, omni)
    rmse_wsa_def  = np.sqrt(np.mean((v_out_default - omni)**2))
    rmse_wsa_fit  = np.sqrt(np.mean((v_out_fitted - omni)**2))
    rmse_pred = np.sqrt(np.mean((v_out_pred - omni)**2))

    # Hard DTW
    dtw_wsa_def  = dtw(v_out_default, omni).normalizedDistance          # distance
    dtw_wsa_fit  = dtw(v_out_fitted, omni).normalizedDistance       # distance
    dtw_pred = dtw(v_out_pred, omni).normalizedDistance             # distance

    # --- Append metrics to CSV ---
    csv_path = os.path.join(PLOTS_DIR, 'metrics.csv')
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'cr',
                'rmse_wsa_def', 'rmse_wsa_fit', 'rmse_pred',
                'mae_wsa_def', 'mae_wsa_fit', 'mae_pred',
                'cc_wsa_def', 'cc_wsa_fit', 'cc_pred',
                'dtw_wsa_def', 'dtw_wsa_fit', 'dtw_pred'
            ])
        writer.writerow([
            cr,
            rmse_wsa_def, rmse_wsa_fit, rmse_pred,
            mae_wsa_def, mae_wsa_fit, mae_pred,
            cc_wsa_def, cc_wsa_fit, cc_pred,
            dtw_wsa_def, dtw_wsa_fit, dtw_pred
        ])

    # Plot setup
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, width_ratios=[1,2], height_ratios=[1,1], figure=fig)
    cmap = 'viridis'

    # Top-left: polar map WSA+HUX
    ax0 = fig.add_subplot(gs[0,0])
    phi = np.linspace(0, 2*np.pi, v2d_wsa.shape[1], endpoint=False)
    R, Phi = np.meshgrid(np.linspace(0.1,1.0,v2d_wsa.shape[0]), phi, indexing='ij')
    X = R * np.cos(Phi); Y = R * np.sin(Phi)
    pcm0 = ax0.pcolormesh(X, Y, v2d_wsa, shading='auto', vmax=700, cmap=cmap)
    ax0.set_aspect('equal'); ax0.set_xlim(-1,1); ax0.set_ylim(-1,1)
    ax0.set_title(f'CR{cr} WSA+HUX'); fig.colorbar(pcm0, ax=ax0)

    # Bottom-left: polar map NN+HUX
    ax1 = fig.add_subplot(gs[1,0])
    pcm1 = ax1.pcolormesh(X, Y, v2d_pred, shading='auto', vmax=700, cmap=cmap)
    ax1.set_aspect('equal'); ax1.set_xlim(-1,1); ax1.set_ylim(-1,1)
    ax1.set_title(f'CR{cr} NN+HUX'); fig.colorbar(pcm1, ax=ax1)

    # Top-right: time series @0.1AU
    ax2 = fig.add_subplot(gs[0,1])
    t = np.arange(T)
    ax2.plot(t, wsa_default_ts, label='WSA Default', color='C0', linestyle='-')
    ax2.plot(t, wsa_fitted_ts, label='WSA Fitted', color='C2', linestyle='--')
    ax2.plot(t, pred_ts, label='NN', color='C1')
    ax2.set_ylim(250, 850)
    ax2.set_title('Boundary @0.1AU'); ax2.set_xlabel('Index'); ax2.set_ylabel('Speed')
    ax2.legend(ncol=3, loc='upper center', fontsize=7)
    ax2.grid(alpha=0.3)

    # Bottom-right: time series @1AU with metrics
    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(t, v_out_default, label=f'WSA Default \n CC={cc_wsa_def:.2f}\n RMSE={rmse_wsa_def:.1f}\n MAE={mae_wsa_def:.1f}\n DTW={dtw_wsa_def:.1f}', color='C0')
    ax3.plot(t, v_out_fitted, label=f'WSA Fitted \n CC={cc_wsa_fit:.2f}\n RMSE={rmse_wsa_fit:.1f}\n MAE={mae_wsa_fit:.1f}\n DTW={dtw_wsa_fit:.1f}', color='C2', linestyle='--')
    ax3.plot(t, v_out_pred, label=f'NN \n CC={cc_pred:.2f}\n RMSE={rmse_pred:.1f}\n MAE={mae_pred:.1f})\n DTW={dtw_pred:.1f}', color='C1')
    ax3.plot(t, omni, label='OMNI', color='black', alpha=0.6)
    ax3.set_ylim(250, 1050)
    ax3.set_title('Extrapolated @1AU vs OMNI'); ax3.set_xlabel('Index'); ax3.set_ylabel('Speed')
    ax3.legend(ncol=4, loc='upper center', fontsize=7)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f'CR{cr}.png'), dpi=100)
    plt.close(fig)
