import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from config_fit import DATA_MANIFEST, EARTH_LAT_IDX, DEVICE

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

        # Load NPZ with feature maps, WSA speed map, and cr_info
        data = np.load(entry['map_path'], allow_pickle=True)
        cr = int(data['cr_info'].item()['cr'])

        # Stack feature channels: expansion factor and min distance
        feats = np.stack([
            data['exp_factors'],
            data['min_distances']
        ], axis=0)  # shape [2, H, W]

        map_tensor = torch.from_numpy(feats).float()

        # Fixed min-max normalization with predefined ranges
        C, H, W = map_tensor.shape
        flat = map_tensor.view(C, -1)

        # Define fixed min-max values for each channel
        fixed_min = torch.tensor([[-4.0], [0.0]], device=flat.device)  # [C, 1]
        fixed_max = torch.tensor([[6.0], [25.0]], device=flat.device)  # [C, 1]

        # Apply normalization with fixed values
        map_tensor = ((flat - fixed_min) / (fixed_max - fixed_min)).view(C, H, W)

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

