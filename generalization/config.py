import os
import torch
import numpy as np
import math
from datetime import datetime

# ==== Data Directories ====
# Root directory containing all data subfolders
DATA_ROOT = "/home/prma1294/wsa_surrogate"
MAP_DIR   = os.path.join(DATA_ROOT, "dataset")
OMNI_DIR  = os.path.join(DATA_ROOT, "OMNI_dataset")
EARTH_IDX_FILE = os.path.join(DATA_ROOT, "earth_lat_idx.pt")

# New directory for generated 2D maps
RESULTS_DIR = "../optimization/results"
WSA_MAPS_DIR = os.path.join(RESULTS_DIR, "wsa_maps")

# ==== Device Configuration ====
# Automatically choose GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Training Hyperparameters ====
# Data
BATCH_SIZE   = 8
VAL_SPLIT    = 0.2
NUM_WORKERS  = 4
PIN_MEMORY   = True      # for DataLoader (faster data loading)

# Optimization
LR           = 1e-5
EPOCHS       = 300
WEIGHT_DECAY = 1e-3

# Loss weights
GAMMA        = 0.1        # Soft‐DTW smoothing parameter

# Misc
SEED         = 45
LOG_FILE     = 'train_log.csv'
CHECKPOINT_DIR = 'checkpoints'

# ==== Load Precomputed Earth‐Latitude Indices ====
# Dict[int CR -> LongTensor of shape [T]]
EARTH_LAT_IDX = torch.load(EARTH_IDX_FILE, map_location='cpu')

# Hard-coded list of CRs to exclude
exclude_crs = [2242, 2245, 2246, 2251, 2265, 2266, 2267, 2275, 2277, 2285, 2286]

# CR extracted at load time from NPZ
DATA_MANIFEST = []
for fname in sorted(os.listdir(MAP_DIR)):
    if not fname.endswith('.npz'):
        continue
    map_path = os.path.join(MAP_DIR, fname)
    data = np.load(map_path, allow_pickle=True)
    cr_info = data['cr_info'].item()
    cr = int(cr_info['cr'])

    # Skip excluded CRs
    if cr in exclude_crs:
        continue

    # Parse start/end times (ISO strings) and compute omega per CR
    t0 = datetime.fromisoformat(cr_info['t_start'])
    t1 = datetime.fromisoformat(cr_info['t_end'])
    period_sec = (t1 - t0).total_seconds()
    omega = 2 * math.pi / period_sec  # rad/s for this CR

    # Corresponding OMNI .txt file
    omni_fname = f"CR{cr}_filtered_OMNI_360.txt"
    omni_path = os.path.join(OMNI_DIR, omni_fname)
    if not os.path.exists(omni_path):
        raise FileNotFoundError(f"OMNI file not found for CR {cr}: {omni_path}")

    # DATA_MANIFEST.append({
    #     'map_path':  map_path,
    #     'omni_path': omni_path,
    #     'omega':     omega
    # })

    # Path to WSA-generated velocity map
    wsap_map_path = os.path.join(WSA_MAPS_DIR, f"fittedCR{cr}_wsa_map.npy")
    if not os.path.exists(wsap_map_path):
        print(f"Warning: WSA map file not found for CR {cr}: {wsap_map_path}")
        continue

    DATA_MANIFEST.append({
        'map_path': map_path,
        'omni_path': omni_path,
        'omega': omega,
        'wsap_map_path': wsap_map_path,
        'cr': cr  # Store CR number explicitly for easier reference
    })

# ==== Summary ====
print(f"Loaded {len(DATA_MANIFEST)} Carrington rotations with per-CR omega.")