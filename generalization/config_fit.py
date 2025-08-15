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

# ==== Device Configuration ====
# Automatically choose GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Precomputed Earth‐Latitude Indices ====
# Dict[int CR -> LongTensor of shape [T]]
EARTH_LAT_IDX = torch.load(EARTH_IDX_FILE, map_location='cpu', weights_only=True)

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

    DATA_MANIFEST.append({
        'map_path':  map_path,
        'omni_path': omni_path,
        'omega':     omega
    })

# ==== Summary ====
print(f"Loaded {len(DATA_MANIFEST)} Carrington rotations with per-CR omega.")