import torch
import torch.nn as nn
from pysdtw import SoftDTW
from hux import HUXPropagator
import torch.nn.functional as F
from config_fit import DEVICE

class SequenceLoss(nn.Module):
    """
    Sequence loss comparing OMNI time series to HUX-propagated speeds.

    Steps:
      1. Extract boundary speeds from 2D map via lat_idx
      2. Propagate via HUX propagator
      3. Compute combined loss: Soft-DTW + RMSE + (1 - PCC)
    """
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.hux = HUXPropagator()
        # Use explicit distance function (from example on website)
        self.soft_dtw = SoftDTW(gamma=gamma, use_cuda=True)  # Use CUDA version

    def _compute_pcc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute 1 - PCC as a loss (lower PCC = higher loss)
        pred, target: [B, T]
        """
        # Center the sequences
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        
        # Compute correlation coefficient for each batch
        numerator = (pred_centered * target_centered).sum(dim=1)
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
        target_std = torch.sqrt((target_centered ** 2).sum(dim=1))
        
        # Avoid division by zero
        denominator = pred_std * target_std
        pcc = numerator / denominator
        
        # Return 1 - mean(PCC) as loss
        return 1.0 - pcc.mean()

    def forward(
        self,
        pred_map: torch.Tensor,
        lat_idx: torch.LongTensor,
        omega: torch.Tensor,
        omni_ts: torch.Tensor
        ) -> torch.Tensor:

        # pred_map: [B,1,H,W]
        B, _, H, W = pred_map.shape

        # Squeeze to [B,H,W]
        v_map = pred_map.squeeze(1)
        T = lat_idx.size(1)

        # Gather boundary speeds: v_in [B,T]
        batch_idx = torch.arange(B, device=v_map.device).unsqueeze(1).expand(B, T)
        time_idx  = torch.arange(T, device=v_map.device).unsqueeze(0).expand(B, T)
        v_in = v_map[batch_idx, time_idx, lat_idx]

        # Propagate to 1 AU: v_out [B,T]
        v_out = self.hux(v_in, omega)

        # Debugging for Nan in speed
        if torch.isnan(omni_ts).any():
            print("\nWarning: omni_ts contains NaN values!")
        if torch.isnan(v_out).any():
            print("\nWarning: v_out contains NaN values!")
            #print(f"v_out: {v_out}")

        # Min-Max scaling for normalization
        v_out_scaled = (v_out - 200) / 1000.0                 # Scaling can be of tanh
        omni_ts_scaled = (omni_ts - 200) / 1000.0

        # 1. Soft-DTW Loss with downsampling
        v_out_dtw = v_out_scaled.unsqueeze(-1)      # [B, T, 1]
        omni_ts_dtw = omni_ts_scaled.unsqueeze(-1)  # [B, T, 1]

        dtw_raw = self.soft_dtw(omni_ts_dtw, v_out_dtw)
        dtw_loss = torch.abs(dtw_raw).mean()   # Scale down the loss for similar order with other losses

        # 2. RMSE Loss
        rmse_loss = torch.sqrt(torch.sum((v_out_scaled - omni_ts_scaled) ** 2))
        #print(f"RMSE Loss: {rmse_loss.item():.4f}")
        
        # 3. Pearson Correlation Coefficient Loss (1 - PCC)
        pcc_loss = self._compute_pcc_loss(v_out_scaled, omni_ts_scaled)
        #print(f"PCC Loss: {pcc_loss.item():.4f}")
 
        # 4. Mean Absolute Error (MAE) Loss
        mae_loss = torch.sum(torch.abs(v_out_scaled - omni_ts_scaled))
        # print(f"MAE Loss: {mae_loss.item():.4f}")

        #print(f"DTW Loss: {dtw_loss.item():.4f}, MAE loss: {mae_loss.item():.4f}, RMSE Loss: {rmse_loss.item():.4f}, PCC Loss: {pcc_loss.item():.4f}")

        return v_out, mae_loss, pcc_loss, dtw_loss, rmse_loss
