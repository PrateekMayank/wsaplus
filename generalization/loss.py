import torch
import torch.nn as nn
from pysdtw import SoftDTW
from hux import HUXPropagator
import pandas as pd
from config import DEVICE

# Load fitted parameters
fitted_params_df = pd.read_csv('../optimization/results/fitted_wsa_params.csv')

# For a specific CR, extract parameters
def get_wsa_params(cr_batch):
    """
    Get WSA parameters for a batch of Carrington Rotations.
    
    Args:
        cr_batch: PyTorch tensor of shape [batch_size] containing CR numbers
        
    Returns:
        Dictionary of parameter tensors, each with shape [batch_size]
    """
    batch_size = len(cr_batch)

    # Initialize parameter tensors
    vmin = torch.zeros(batch_size, device=DEVICE)
    vmax = torch.zeros(batch_size, device=DEVICE)
    alpha = torch.zeros(batch_size, device=DEVICE)
    beta = torch.zeros(batch_size, device=DEVICE)
    a1 = torch.zeros(batch_size, device=DEVICE)
    a2 = torch.zeros(batch_size, device=DEVICE)
    w = torch.zeros(batch_size, device=DEVICE)

    # Fill in parameters for each CR in batch
    for i, cr_num in enumerate(cr_batch):
        # Convert tensor to CPU scalar before comparison
        if isinstance(cr_num, torch.Tensor):
            cr_value = cr_num.cpu().item()
        else:
            cr_value = cr_num
        # Find parameters for this CR
        cr_params = fitted_params_df[fitted_params_df['CR'] == cr_value]

        # Extract parameters
        vmin[i] = cr_params['vmin'].values[0]
        vmax[i] = cr_params['vmax'].values[0]
        alpha[i] = cr_params['alpha'].values[0]
        beta[i] = cr_params['beta'].values[0]
        a1[i] = cr_params['a1'].values[0]
        a2[i] = cr_params['a2'].values[0]
        w[i] = cr_params['w'].values[0]
    
    return {
            'vmin': vmin,
            'vmax': vmax,
            'alpha': alpha,
            'beta': beta,
            'a1': a1,
            'a2': a2,
            'w': w
    }

class MapLoss(nn.Module):
    """
    Pixel-wise MSE loss between predicted and reference speed maps at 0.1 AU.

    pred_map: [B,1,H,W] or [B,H,W]
    target_map: [B,H,W]
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_map: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
        # Squeeze channel dimension if needed
        if pred_map.dim() == 4 and pred_map.size(1) == 1:
            pred = pred_map.squeeze(1)
        else:
            pred = pred_map

        wsa_loss = self.mse(pred, target_map.to(pred.device))

        return wsa_loss

class WSAResidualLoss(nn.Module):
    """
    Physics-based residual loss based on WSA equation.
    Enforces the relationship between expansion factor, magnetogram angle, and solar wind speed.
    
    input_map[0] = exp_factor
    input_map[1] = magnetogram angle deviation (mad)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_map, input_map, cr):
        # Extract required inputs
        exp_factor = input_map[:, 0, :, :]      # [B, H, W]
        mad = input_map[:, 1, :, :]             # [B, H, W]
        batch_size = input_map.size(0)

        # Squeeze channel dimension if needed
        if pred_map.dim() == 5 and pred_map.size(1) == 1:
            pred = pred_map.squeeze(1)
        else:
            pred = pred_map
        
        # Denormalize channel values
        exp_factor_denorm = exp_factor * 10.0 - 4.0     # (max-min)=10, min=-4
        mad_denorm = mad * 25.0                         # (max-min)=25, min=0
        mad_denorm = mad_denorm * (torch.pi / 180.0)    # Convert degrees to radians

        # WSA equation calculation
        # Load optmized parameters values
        params = get_wsa_params(cr)
        params_size = len(params['vmin'])

        # Reshape parameter tensors for proper broadcasting
        vmin = params['vmin'].view(batch_size, 1, 1)   # [8] -> [8, 1, 1] 
        vmax = params['vmax'].view(batch_size, 1, 1)   # [8] -> [8, 1, 1]
        alpha = params['alpha'].view(batch_size, 1, 1) # [8] -> [8, 1, 1]
        beta = params['beta'].view(batch_size, 1, 1)   # [8] -> [8, 1, 1]
        a1 = params['a1'].view(batch_size, 1, 1)       # [8] -> [8, 1, 1]
        a2 = params['a2'].view(batch_size, 1, 1)       # [8] -> [8, 1, 1]
        w = params['w'].view(batch_size, 1, 1)         # [8] -> [8, 1, 1]

        power_term = 1 + torch.pow(10, exp_factor_denorm)
        term1 = vmax / torch.pow(power_term, alpha)

        exp_term = torch.exp(-1 * torch.pow((mad_denorm / w), beta))
        term2 = torch.pow(1.0 - a1 * exp_term, a2)

        # Print intermediate values to help debug
        if torch.isnan(term1).any() or torch.isnan(term2).any():
            print(f"NaN detected! term1_min: {term1.min().item()}, term1_max: {term1.max().item()}")
            print(f"term2_min: {term2.min().item()}, term2_max: {term2.max().item()}")
    
        wsa_speed = vmin + term1 * term2

        # Compute residual loss
        residual = torch.abs(pred - wsa_speed)

        loss = torch.mean(residual) # ** 2)  # MSE
        
        return loss