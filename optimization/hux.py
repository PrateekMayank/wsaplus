import torch
import torch.nn as nn
import math

class HUXPropagator(nn.Module):
    """
    Differentiable HUX propagator mapping boundary speeds at r_initial to speeds at r_final.

    Inputs:
      v_in: Tensor of shape [B, T], speeds at inner boundary (0.1 AU) for T longitudes/time steps
      omega: Tensor of shape [B] or scalar, solar rotation rate (rad/s) for each batch

    Output:
      v_out: Tensor of shape [B, T], speeds at outer boundary (1.0 AU), flipped for time alignment
    """
    def __init__(self, r_initial=0.1, r_final=1.0, n_radius=512, au_km=1.496e8):
        super().__init__()
        self.r_initial = r_initial
        self.r_final   = r_final
        self.n_radius  = n_radius

        # Radial step in kilometers
        dr_AU = (r_final - r_initial) / (n_radius - 1)
        dr_km = dr_AU * au_km

        self.register_buffer('dr_km', torch.tensor(dr_km, dtype=torch.float32))

    def forward(self, v_in: torch.Tensor, omega) -> torch.Tensor:
        """
        Perform HUX propagation.

        Args:
            v_in (Tensor): [B, T]
            omega (Tensor or float): [B] or scalar
        Returns:
            v_out (Tensor): [B, T] at r_final, flipped along T for time alignment
        """
        B, T = v_in.shape
        device = v_in.device

        # Compute angular step
        dphi = 2 * math.pi / T

        # Start with the inner boundary v_prev = v_in
        v_prev = v_in

        # Build omega tensor [B]
        if not torch.is_tensor(omega):
            omega = torch.full((B,), omega, device=device, dtype=v_in.dtype)
        omega = omega.view(B, 1)        # now [B,1] for coeff computation, earlier it was [B]

        # Coefficient Δr*Ω/Δφ
        coeff = self.dr_km * omega / dphi  # shape [B,1]

        # Recurrence: update v_prev radius-by-radius
        for _ in range(1, self.n_radius):
            v_next = torch.roll(v_prev, shifts=-1, dims=-1)  # [B, T]
            update = coeff * ((v_next - v_prev) / v_prev)    # [B, T]
            v_prev = v_prev + update                         # produces new tensor

        # Extract outer boundary and flip for time alignment with OMNI
        v_out = torch.flip(v_prev, dims=[1])    # [B, T]

        return v_out
