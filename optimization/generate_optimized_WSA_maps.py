"""
Generate 2D WSA velocity maps using fitted parameters from CSV file
Applies WSA equation to expansion factor and minimum angular distance maps
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from dataset_fit import CarringtonDataset
from config_fit import DATA_MANIFEST
import argparse

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_wsa_equation(exp_factor, mad, params):
    """
    Apply WSA equation with given parameters to compute solar wind speed
    
    Args:
        exp_factor: 2D expansion factor map [H, W]
        mad: 2D minimum angular distance map [H, W]
        params: dict with keys vmin, vmax, alpha, beta, a1, a2
    
    Returns:
        2D velocity map [H, W]
    """
    # Extract parameters
    vmin = params['vmin']
    vmax = params['vmax']
    alpha = params['alpha']
    beta = params['beta']
    a1 = params['a1']
    a2 = params['a2']
    w = params['w']
    
    # Denormalize channel values
    exp_factor_denorm = exp_factor * 10.0 - 4.0     # (max-min)=10, min=-4
    mad_denorm = mad * 25.0                         # (max-min)=25, min=0
    mad_denorm = mad_denorm * (torch.pi / 180.0)    # Convert degrees to radians
    
    # WSA equation calculation
    power_term = 1 + torch.pow(10, exp_factor_denorm)
    term1 = vmax / torch.pow(power_term, alpha)

    exp_term = torch.exp(-1 * torch.pow((mad_denorm / w), beta))
    term2 = torch.pow(1 - a1 * exp_term, a2)

    wsa_speed = vmin + term1 * term2
    
    return wsa_speed

def load_cr_data(entry):
    """Load Carrington Rotation data"""
    ds = CarringtonDataset(manifest=[entry])
    sample = ds[0]
    
    exp_map = sample['map'][0].to(DEVICE)     # Expansion factor map
    d_map = sample['map'][1].to(DEVICE)       # Minimum angular distance map
    cr = sample['cr']                         # Carrington Rotation number
    
    return cr, exp_map, d_map

def visualize_map(v_map, cr, output_dir, prefix="fitted"):
    """Create visualization of velocity map"""
    plt.figure(figsize=(12, 6))
    
    # Convert tensor to numpy for plotting
    v_map_np = v_map.cpu().detach().numpy()

    
    # Plot using pcolormesh
    im = plt.imshow(v_map_np.T, origin='lower', cmap='plasma', interpolation='nearest')
    
    plt.colorbar(im, label='Solar Wind Speed (km/s)')
    plt.title(f'CR {cr} - WSA Velocity Map')
    plt.xlabel('Carrington Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    
    # Save figure
    filename = os.path.join(output_dir, f'{prefix}CR{cr}_wsa_map.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {filename}")
    
    # Also save as numpy array
    np_filename = os.path.join(output_dir, f'{prefix}CR{cr}_wsa_map.npy')
    np.save(np_filename, v_map_np)
    print(f"Saved numpy array to {np_filename}")

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load fitted parameters
    params_df = pd.read_csv(args.params_file)
    print(f"Loaded {len(params_df)} fitted parameter sets")
    
    # Process each Carrington Rotation
    for i, entry in enumerate(DATA_MANIFEST):
        try:
            # Load data
            cr, exp_map, d_map = load_cr_data(entry)
            
            # Find parameters for this CR
            cr_params = params_df[params_df['CR'] == cr]
            if len(cr_params) == 0:
                print(f"⚠️ No fitted parameters found for CR {cr}, skipping...")
                continue
            
            # Extract parameters
            params = {
                'vmin': cr_params['vmin'].values[0],
                'vmax': cr_params['vmax'].values[0],
                'alpha': cr_params['alpha'].values[0],
                'beta': cr_params['beta'].values[0],
                'a1': cr_params['a1'].values[0],
                'a2': cr_params['a2'].values[0],
                'w': cr_params['w'].values[0]
            }
            
            print(f"Processing CR {cr} with parameters: {params}")
            
            # Apply WSA equation
            v_map = apply_wsa_equation(exp_map, d_map, params)

            # Visualize and save
            visualize_map(v_map, cr, args.output_dir)
            
            print(f"✅ Completed CR {cr}")
            
        except Exception as e:
            import traceback
            print(f"❌ CR {entry} failed:", e)
            print("Traceback:")
            traceback.print_exc()
            continue
            
    print(f"All processing complete. Maps saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate WSA velocity maps from fitted parameters')
    parser.add_argument('--params-file', type=str, default='results/fitted_wsa_params.csv',
                        help='Path to fitted parameters CSV file')
    parser.add_argument('--output-dir', type=str, default='results/wsa_maps',
                        help='Directory to save generated maps')
    args = parser.parse_args()
    
    main(args)