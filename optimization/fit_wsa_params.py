"""
WSA Parameter Fitting Script

This script fits WSA (Wang-Sheeley-Arge) equation parameters for each Carrington Rotation (CR)
by optimizing the parameters to best match OMNI in-situ solar wind speed observations.

The optimization uses multiple loss components:
- Mean Absolute Error (MAE)
- Pearson Correlation Coefficient (PCC)
- Dynamic Time Warping (DTW)
- Root Mean Square Error (RMSE)
- Trend matching (first derivative)
- Curvature matching (second derivative)

Output:
- fitted_wsa_params.csv: Optimized parameters for each CR
- fitted_wsa_metrics.csv: Performance metrics for fitted vs default parameters
- CR{cr}_comparison.png: Visualization plots for each CR
"""

import os
import csv
import torch
import torch.nn as nn
from torch.optim import LBFGS
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataset_fit import CarringtonDataset
from loss_fit import SequenceLoss
from config_fit import DATA_MANIFEST, DEVICE

# WSA parameter ranges for optimization
PARAM_RANGES = {
    'vmin': {'default': 275.0, 'min': 250.0, 'max': 300.0},    # Minimum solar wind speed (km/s)
    'vmax': {'default': 750.0, 'min': 600.0, 'max': 900.0},    # Maximum solar wind speed (km/s)
    'alpha': {'default': 0.222, 'min': 0.05, 'max': 0.5},      # Expansion factor exponent
    'beta': {'default': 1.50, 'min': 0.5, 'max': 3.0},         # MAD exponent
    'a1': {'default': 0.80, 'min': 0.70, 'max': 0.90},         # MAD amplitude factor
    'a2': {'default': 1.0, 'min': 0.1, 'max': 6.0},            # Term2 exponent
    'w': {'default': 0.028, 'min': 0.01, 'max': 0.05}          # MAD normalization factor
}

# Extract default values for initialization
PARAM_DEFAULTS = {k: v['default'] for k, v in PARAM_RANGES.items()}

# Fixed WSA parameters (for baseline comparison)
WSA_DEFAULTS = {
    'vmin': 250.0,
    'vmax': 750.0, 
    'alpha': 0.222,
    'beta': 1.25,
    'a1': 0.8,
    'a2': 3.0,
    'w': 0.028
}

# Fixed ranges for baseline WSA (no variation)
WSA_RANGES = {param: {'default': val, 'min': val, 'max': val} 
              for param, val in WSA_DEFAULTS.items()}

class WSAParameterFitter(nn.Module):
    """
    Neural network module that holds and optimizes WSA equation parameters.
    
    Uses tanh activation to bound parameters within specified ranges:
    param_value = center + half_range * tanh(raw_parameter)
    """
    
    def __init__(self, defaults, param_ranges):
        super().__init__()
        self.names = list(defaults.keys())
        self.defaults = defaults
        self.param_ranges = param_ranges
        
        # Initialize raw parameters at 0 (tanh(0) = 0 → parameters at center of range)
        self.raw = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(())) for name in self.names
        })

    def forward(self, exp_factor, mad):
        """
        Apply WSA equation with current parameter values.
        
        Args:
            exp_factor: Expansion factor map [B, H, W]
            mad: Minimum angular distance map [B, H, W]
            
        Returns:
            wsa_speed: Computed solar wind speed map [B, H, W]
        """
        # Transform raw parameters to bounded values
        params = {}
        for name in self.names:
            min_val = self.param_ranges[name]['min']
            max_val = self.param_ranges[name]['max']
            center = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            params[name] = center + half_range * torch.tanh(self.raw[name])

        # Denormalize channel values
        exp_factor_denorm = exp_factor * 10.0 - 4.0     # (max-min)=10, min=-4
        mad_denorm = mad * 25.0                         # (max-min)=25, min=0
        mad_denorm = mad_denorm * (torch.pi / 180.0)    # Convert degrees to radians

        # Extract parameter values
        vmin, vmax, alpha, beta, a1, a2, w = (
            params['vmin'], params['vmax'], params['alpha'],
            params['beta'], params['a1'], params['a2'], params['w']
        )
        
        # WSA equation calculation
        power_term = 1 + torch.pow(10, exp_factor_denorm)
        term1 = vmax / torch.pow(power_term, alpha)
        
        exp_term = torch.exp(-1 * torch.pow((mad_denorm / w), beta))
        term2 = torch.pow(1 - a1 * exp_term, a2)
        
        wsa_speed = vmin + term1 * term2
        
        return wsa_speed

    def get_param_vector(self):
        """Return the current actual parameter values as a list"""
        vec = []
        for name in self.names:
            min_val = self.param_ranges[name]['min']
            max_val = self.param_ranges[name]['max']
            center = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            vec.append((center + half_range * torch.tanh(self.raw[name])).item())
        return vec


# ----------------------------------
def load_cr_data(entry):
    """Load data for a single Carrington Rotation."""
    ds = CarringtonDataset(manifest=[entry])
    sample = ds[0]
    
    # Move tensors to device and add batch dimension where needed
    exp_map = sample['map'][0].unsqueeze(0).to(DEVICE)
    d_map = sample['map'][1].unsqueeze(0).to(DEVICE)
    lat_idx = sample['lat_idx'].unsqueeze(0).to(DEVICE)
    omni_ts = sample['omni'].unsqueeze(0).to(DEVICE)
    omega = sample['omega'].unsqueeze(0).to(DEVICE)
    cr = sample['cr']
    
    return cr, exp_map, d_map, lat_idx, omega, omni_ts
# ----------------------------------

def trend_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss between first-order differences (slopes).
    Encourages similar trends between predicted and target sequences.
    """
    d_pred = torch.abs(pred[:, 1:] - pred[:, :-1])
    d_target = torch.abs(target[:, 1:] - target[:, :-1])
    return F.l1_loss(d_pred, d_target)

def curvature_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss between second-order differences (curvature).
    Encourages similar curvature patterns between predicted and target sequences.
    """
    d2_pred = torch.abs(pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2])
    d2_target = torch.abs(target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2])
    return F.l1_loss(d2_pred, d2_target)

class UncertaintyLossWrapper(nn.Module):
    """
    Learns optimal weights for different loss components using uncertainty weighting.
    Each loss is weighted as: exp(-log_var) * loss + log_var
    """
    
    def __init__(self):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(0.0)) 
            for k in ['mae', 'pcc', 'dtw', 'rmse', 'trend', 'curvature']
        })

    def forward(self, losses):
        """Combine losses with learned uncertainty weights."""
        return sum(torch.exp(-self.log_vars[k]) * v + self.log_vars[k] 
                  for k, v in losses.items())

# ------------------------------------

def fit_cr(cr, exp_map, d_map, lat_idx, omega, omni_ts, loss_fn, unc_wrapper, max_iter=50):
    """
    Fit WSA parameters for a single Carrington Rotation.
    
    Args:
        cr: Carrington Rotation number
        exp_map, d_map: Input feature maps
        lat_idx: Earth latitude indices for boundary extraction
        omega: Solar rotation rate
        omni_ts: OMNI solar wind speed observations
        loss_fn: Loss function for evaluation
        unc_wrapper: Uncertainty weighting for loss combination
        max_iter: Maximum optimization iterations
        
    Returns:
        row_params: [CR, param1, param2, ...] for CSV output
        row_metrics: Dictionary of performance metrics
    """
    # Initialize parameter fitter and optimizer
    fitter = WSAParameterFitter(PARAM_DEFAULTS, PARAM_RANGES).to(DEVICE)
    optimizer = LBFGS(fitter.parameters(), lr=0.5, max_iter=100, 
                     line_search_fn='strong_wolfe')
    step_counter = [0]

    def closure():
        """Optimization closure function for LBFGS."""
        optimizer.zero_grad()
        
        # Forward pass and compute raw losses
        v_map = fitter(exp_map, d_map).unsqueeze(0)
        v_model, mae_loss, pcc_loss, dtw_loss, rmse_loss = loss_fn(
            v_map, lat_idx, omega, omni_ts)
        t_loss = trend_loss(v_model, omni_ts)
        c_loss = curvature_loss(v_model, omni_ts)
        
        # Apply manual loss weights for better optimization
        mae_loss = 0.1 * t_loss * mae_loss
        pcc_loss = 50 * pcc_loss
        dtw_loss = 0.1 * dtw_loss
        rmse_loss = t_loss * rmse_loss
        t_loss = 1.5 * t_loss
        c_loss = 1.5 * c_loss
        
        # Combine losses using uncertainty weighting
        loss = unc_wrapper({
            'mae': mae_loss, 'pcc': pcc_loss, 'dtw': dtw_loss, 
            'rmse': rmse_loss, 'trend': t_loss, 'curvature': c_loss
        })
        
        # Print progress
        print(f"    Step {step_counter[0]} losses -> "
              f"MAE: {mae_loss.item():.2f}, PCC: {pcc_loss.item():.2f}, "
              f"DTW: {dtw_loss.item():.2f}, RMSE: {rmse_loss.item():.2f}, "
              f"Total: {loss.item():.2f}, Trend: {t_loss.item():.4f}, "
              f"Curvature: {c_loss.item():.4f}")
        
        loss.backward()
        step_counter[0] += 1
        return loss

    # Run optimization
    while step_counter[0] < max_iter:
        loss = optimizer.step(closure)
        if step_counter[0] % 50 == 0:
            print(f"    CR {cr} — iter {step_counter[0]}/{max_iter}, loss = {loss:.4f}")

    # Evaluate final performance
    with torch.no_grad():
        # Fitted parameters performance
        v_map = fitter(exp_map, d_map).unsqueeze(0)
        v_model, mae_loss, pcc_loss, dtw_loss, rmse_loss = loss_fn(
            v_map, lat_idx, omega, omni_ts)
        
        # Default parameters performance (for comparison)
        default_fitter = WSAParameterFitter(WSA_DEFAULTS, WSA_RANGES).to(DEVICE)
        v_map_default = default_fitter(exp_map, d_map).unsqueeze(0)
        v_model_default, mae_loss_d, pcc_loss_d, dtw_loss_d, rmse_loss_d = loss_fn(
            v_map_default, lat_idx, omega, omni_ts)
        
        # Convert to numpy for plotting
        v_np = v_model.cpu().numpy().ravel()
        omni_np = omni_ts.cpu().numpy().ravel()
        v_default_np = v_model_default.cpu().numpy().ravel()


    # Create comparison plot
    tuned_str = (f"MAE={mae_loss.item():.2f}, PCC={(1-pcc_loss).item():.2f}, "
                f"DTW={dtw_loss.item():.2f}, RMSE={rmse_loss.item():.2f}")
    default_str = (f"MAE={mae_loss_d.item():.2f}, PCC={(1-pcc_loss_d).item():.2f}, "
                  f"DTW={dtw_loss_d.item():.2f}, RMSE={rmse_loss_d.item():.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(v_np, label="Fitted WSA", color="C0")
    plt.plot(v_default_np, label="Default WSA", linestyle="--", color="C1")
    plt.plot(omni_np, label="OMNI (observed)", color="k")
    plt.ylim(250, 750)
    plt.xlabel("Time Step")
    plt.ylabel("Speed (km/s)")
    plt.title(f"CR {cr}\nTuned Losses: {tuned_str}\nDefault Losses: {default_str}", 
             fontsize=10)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(f"results/CR{cr}_comparison.png")
    plt.close()

    # Prepare output data
    row_params = [cr] + fitter.get_param_vector()
    row_metrics = {
        "mae": mae_loss.item(),
        "pcc": (1 - pcc_loss).item(),
        "dtw": dtw_loss.item(),
        "rmse": rmse_loss.item(),
        "mae_d": mae_loss_d.item(),
        "pcc_d": (1 - pcc_loss_d).item(),
        "dtw_d": dtw_loss_d.item(),
        "rmse_d": rmse_loss_d.item(),
    }

    return row_params, row_metrics
# ----------------------------------

def main():
    """Main function to fit WSA parameters for all Carrington Rotations."""
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize output files and loss functions
    out_path = os.path.join('results', 'fitted_wsa_params.csv')
    metrics_path = os.path.join('results', 'fitted_wsa_metrics.csv')
    loss_fn = SequenceLoss().to(DEVICE)
    unc_wrapper = UncertaintyLossWrapper().to(DEVICE)

    # Process all Carrington Rotations
    with open(out_path, 'w', newline='') as params_file, \
         open(metrics_path, 'w', newline='') as metrics_file:

        params_writer = csv.writer(params_file)
        metrics_writer = csv.writer(metrics_file)

        # Write CSV headers
        params_writer.writerow(['CR'] + list(PARAM_DEFAULTS.keys()))
        metrics_writer.writerow([
            'CR', 'MAE', 'PCC', 'DTW', 'RMSE', 'MAE_d', 'PCC_d', 'DTW_d', 'RMSE_d'
        ])

        # Fit parameters for each CR
        for entry in DATA_MANIFEST:
            try:
                cr, exp_map, d_map, lat_idx, omega, omni_ts = load_cr_data(entry)
                print(f"➡️  Fitting CR {cr} ...")
                
                row_params, row_metrics = fit_cr(
                    cr, exp_map, d_map, lat_idx, omega, omni_ts, loss_fn, unc_wrapper
                )
                
                # Write results to CSV files
                params_writer.writerow(row_params)
                metrics_writer.writerow([
                    cr, row_metrics["mae"], row_metrics["pcc"], row_metrics["dtw"],
                    row_metrics["rmse"], row_metrics["mae_d"], row_metrics["pcc_d"],
                    row_metrics["dtw_d"], row_metrics["rmse_d"]
                ])
                
                print(f"✅ Completed CR {cr}")
                
            except Exception as e:
                import traceback
                print(f"❌ CR {entry} failed: {e}")
                print("Traceback:")
                traceback.print_exc()
                continue

    print(f"Parameter fitting complete! Results saved to {out_path} and {metrics_path}")


if __name__ == '__main__':
    main()
