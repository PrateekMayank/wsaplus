import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import CarringtonDataset
from model import WSASurrogateModel
from config import DEVICE, CHECKPOINT_DIR, DATA_MANIFEST

def apply_wsa_equation(exp_factor, mad):
    """Apply WSA equation with given parameters"""
    # WSA parameters
    vmin = 250
    vmax = 750
    alpha = 0.222
    beta = 1.25
    a1 = 0.8
    a2 = 3.0
    w = 0.028
    
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

def plot_combined_3cr_figure():
    """
    Create a combined figure with 3 specific CRs from validation dataset
    Each row shows: WSA Default, WSA Optimized, WSA+, and Text Summary
    """
    # Load model
    model = WSASurrogateModel().to(DEVICE)
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'last.pt'), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Load validation dataset (every 6th sample)
    val_manifest = [DATA_MANIFEST[i] for i in range(len(DATA_MANIFEST)) if i % 6 == 0]
    val_ds = CarringtonDataset(manifest=val_manifest)
    print(f"Validation CRs: {[val_ds[i]['cr'] for i in range(len(val_ds))]}")
    
    # Target CRs for plotting
    target_crs = [2049, 2080, 2161]
    #target_crs = [2072, 2094, 2155]
    
    # Find indices for target CRs in validation dataset
    cr_indices = {}
    for idx in range(len(val_ds)):
        cr_num = val_ds[idx]['cr']
        if cr_num in target_crs:
            cr_indices[cr_num] = idx
    
    # Check if all target CRs are found
    missing_crs = [cr for cr in target_crs if cr not in cr_indices]
    if missing_crs:
        print(f"Warning: CRs {missing_crs} not found in validation dataset")
        print(f"Available CRs: {[val_ds[i]['cr'] for i in range(len(val_ds))]}")
        return
    
    # Create figure with custom column widths - 4th column is narrower
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), 
                           gridspec_kw={'width_ratios': [1, 1, 1, 0.5]})
    
    # Define row labels and common settings
    row_labels = ['a', 'b', 'c']
    speed_levels = np.linspace(300, 800, 10)
    
    for row, cr_num in enumerate(target_crs):
        if cr_num not in cr_indices:
            continue
            
        idx = cr_indices[cr_num]
        sample = val_ds[idx]
        
        # Get sample data
        x_map = sample['map'].unsqueeze(0).to(DEVICE)
        wsa_optimized = sample['wsap'].cpu().numpy()
        
        # Get model prediction (WSA+)
        with torch.no_grad():
            wsa_plus = model(x_map)[0, 0].cpu().numpy()
        
        # Extract input channels for WSA default calculation
        input_map = sample['map'].cpu().numpy()
        exp_factor = input_map[0]  # First channel: expansion factor
        mad = input_map[1]         # Second channel: MAD
        
        # Calculate WSA Default
        wsa_default = apply_wsa_equation(exp_factor, mad)
        
        # Calculate differences for statistics
        diff_default = wsa_plus - wsa_default
        diff_optimized = wsa_plus - wsa_optimized
        
        # Define coordinate grids
        lon = np.linspace(0, 360, wsa_default.shape[0])
        lat = np.linspace(-1, 1, wsa_default.shape[1])  # sine of latitude in degrees

        LON, LAT = np.meshgrid(lon, lat)
        
        # Data and titles for 3 subplots per row
        plots_data = [
            (wsa_default, f'{row_labels[row]}1) CR{cr_num} WSA Default', 'viridis', speed_levels, 'Solar Wind Speed (km/s)'),
            (wsa_optimized, f'{row_labels[row]}2) CR{cr_num} WSA Optimized', 'viridis', speed_levels, 'Solar Wind Speed (km/s)'),
            (wsa_plus, f'{row_labels[row]}3) CR{cr_num} WSA+', 'viridis', speed_levels, 'Solar Wind Speed (km/s)')
        ]
        
        # Plot first 3 panels for this CR
        for col, (data, title, cmap, levels, cbar_label) in enumerate(plots_data):
            ax = axs[row, col]
            
            # Create contour plot with arcsin latitude grid
            im = ax.contourf(LON, LAT, data.T, levels=levels, cmap=cmap, extend='both')
            
            # Formatting
            ax.set_title(title, fontsize=16)

            # Set axis labels - matching your coordinate system
            if row == 2:  # Only bottom row gets x-axis labels
                ax.set_xlabel('Longitude (°)', fontsize=15)
            
            if col == 0:  # Only leftmost column gets y-axis labels
                ax.set_ylabel('Sine of Latitude (°)', fontsize=15)
            
            # Set axis limits for arcsin grid
            ax.set_xlim(0, 360)
            ax.set_ylim(-1, 1)  # Range of arcsin(-1) to arcsin(1)
            
            # Set ticks with appropriate labels
            ax.set_xticks(np.arange(0, 361, 90))
            
            # Create custom y-ticks at arcsin positions
            lat_ticks = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_yticks(lat_ticks)
            ax.minorticks_on()
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            #cbar.set_label(cbar_label, fontsize=12)
            
            # Set colorbar ticks
            speed_ticks = np.linspace(300, 800, 5)
            cbar.set_ticks(speed_ticks)
            cbar.set_ticklabels([f'{int(tick)}' for tick in speed_ticks])
            cbar.ax.tick_params(labelsize=12)
        
        # Fourth panel: Text summary
        ax_text = axs[row, 3]
        ax_text.axis('off')
        ax_text.set_title(f'{row_labels[row]}4) Statistical Summary', fontsize=15)
        
        # Calculate comprehensive statistics
        stats_summary = f"""
CR{cr_num} Analysis
{'='*25}

RMSE Comparison:
vs Default: {np.sqrt(np.mean(diff_default**2)):.1f} km/s
vs Optimized: {np.sqrt(np.mean(diff_optimized**2)):.1f} km/s

Correlation Comparision:
vs Default: {np.corrcoef(wsa_plus.flatten(), wsa_default.flatten())[0,1]:.3f}
vs Optimized: {np.corrcoef(wsa_plus.flatten(), wsa_optimized.flatten())[0,1]:.3f}
"""
        
        # Add text to the panel - adjusted for narrower column
        ax_text.text(-0.03, 0.85, stats_summary, transform=ax_text.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Create output directory and save
    os.makedirs('CR_metrics', exist_ok=True)
    save_path = 'CR_metrics/2D_maps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined 3-CR validation plot: {save_path}")
    
    # Print summary statistics
    for cr_num in target_crs:
        if cr_num in cr_indices:
            print(f"CR {cr_num}: Plotted successfully")

if __name__ == "__main__":
    print("Creating combined 3-CR validation figure...")
    plot_combined_3cr_figure()
    print("Done!")