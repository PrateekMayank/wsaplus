import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from config import DATA_MANIFEST, EARTH_LAT_IDX, DEVICE

class CarringtonDataset(Dataset):
    """
    PyTorch Dataset for Carrington rotation feature maps and OMNI time series.

    Expects DATA_MANIFEST as a list of dicts with keys:
      - 'map_path': str, path to NPZ file containing feature grids and WSA speed
      - 'omni_path': str, path to .txt file with OMNI time series
      - 'omega': float, Carrington rotation rate (rad/s)

    EARTH_LAT_IDX: dict mapping CR -> LongTensor of shape [T] with precomputed
                   indices for sampling Earth latitudes from feature maps.
    """
    def __init__(self, manifest=DATA_MANIFEST):
        super().__init__()
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]

        # Extract the CR number from the entry
        cr = entry['cr']

        # Load NPZ with feature maps, WSA speed map, and cr_info
        data = np.load(entry['map_path'], allow_pickle=True)
        exp_map = torch.tensor(data['exp_factors'], dtype=torch.float32)   # [H, W]
        mad_map = torch.tensor(data['min_distances'], dtype=torch.float32)           # [H, W]
        #magnetogram = torch.tensor(data['br_solar'], dtype=torch.float32)       # [H, W]

        # Stack channels
        map_tensor = torch.stack([exp_map, mad_map], dim=0)        # [2/3, H, W]

        # Load pre-generated WSA map from npy file instead of calculating it
        wsap_map = torch.tensor(np.load(entry['wsap_map_path']), dtype=torch.float32)  # [H, W]


        # Fixed min-max normalization with predefined ranges
        C, H, W = map_tensor.shape

        # Split channels for separate normalization
        exp_map = map_tensor[0].view(-1)  # Expansion factor
        mad_map = map_tensor[1].view(-1)  # Minimum angular distance
        #mag_map = map_tensor[2].view(-1)  # Magnetogram

        # Apply min-max normalization to first two channels
        exp_min, exp_max = -4.0, 6.0
        mad_min, mad_max = 0.0, 25.0

        exp_norm = (exp_map - exp_min) / (exp_max - exp_min)
        mad_norm = (mad_map - mad_min) / (mad_max - mad_min)

        # Apply tanh normalization to magnetogram
        #scale_factor = 5.0  # Adjust based on your magnetogram data distribution
        #mag_norm = torch.tanh(mag_map / scale_factor)  # Maps to [-1, 1] range

        # If you want to rescale from [-1, 1] to [0, 1]:
        #mag_norm = (mag_norm + 1) / 2

        # Recombine the normalized channels
        map_tensor = torch.stack([
            exp_norm.view(H, W),
            mad_norm.view(H, W)#,
            #mag_norm.view(H, W)
        ], dim=0)

        if torch.isnan(map_tensor).any() or torch.isinf(map_tensor).any():
            raise RuntimeError(f"Map {entry['map_path']} contains NaN/Inf at load time!")

        # Load mandatory WSA reference speed map [H, W]
        wsa_speed = torch.from_numpy(data['speed']).float()     # CPU tensor

        # Load OMNI time series [T]
        omni = self._load_omni(entry['omni_path'])  # CPU tensor

        # Precomputed Earth-latitude indices for this CR [T]
        lat_idx = EARTH_LAT_IDX[cr]     # LongTensor on CPU

        # Carrington rotation rate for this sample [scalar]
        omega = torch.tensor(entry['omega'], dtype=torch.float32)   # scalar CPU tensor

        return {
            'cr':      cr,
            'map':     map_tensor,  # [2, H, W] on CPU
            'wsap':    wsap_map,    # [H, W] on CPU
            'wsa':     wsa_speed,   # [H, W] on CPU
            'omni':    omni,        # [T] on CPU
            'lat_idx': lat_idx,     # [T] on CPU
            'omega':   omega        # scalar on CPU
        }


    @staticmethod
    def _load_omni(path):
        """
        Load OMNI time series from file, return torch.Tensor [T] on CPU.
        Automatically handles headers and finds a column containing 'speed'.
        """
        # Read data using pandas and extract bulk_speed column
        df = pd.read_csv(path, sep="\t")

        speeds = pd.to_numeric(df["bulk_speed"], errors="coerce")  
        
        # Convert to float32 tensor
        return torch.tensor(speeds.values, dtype=torch.float32)

