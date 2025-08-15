import os
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw


def plot_map_comparison(pred_map, wsa_map, sample_id, epoch, save_dir='plots'):
    """
    Plot predicted vs reference speed maps and percent difference.

    Args:
        pred_map (np.ndarray): [H, W] predicted 2D speed map
        wsa_map (np.ndarray):  [H, W] reference WSA speed map
        sample_id (int):       sample identifier
        epoch (int):           training epoch
        save_dir (str):        directory to save plots
    """

    diff = (pred_map - wsa_map) / (wsa_map) * 100.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axes[0].imshow(pred_map, origin='lower', cmap='plasma', vmin=300, vmax=800)
    axes[0].set_title('Predicted')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(wsa_map, origin='lower', cmap='plasma', vmin=300, vmax=800)
    axes[1].set_title('WSA Reference')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu', vmin=-60, vmax=60)
    axes[2].set_title('Percent Difference')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f'Sample {sample_id} - Epoch {epoch}', fontsize=16)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'map_cmp_s{sample_id}_ep{epoch}.png'), dpi=150)
    plt.close(fig)


def plot_dtw_alignment(v_extrap, omni_ts, sample_id, epoch, save_dir='plots'):
    """
    Plot time series comparison and DTW alignment between extrapolated and OMNI speeds.

    Args:
        v_extrap (np.ndarray): [T] extrapolated speed series
        omni_ts  (np.ndarray): [T] observed OMNI speed series
        sample_id (int):       sample identifier
        epoch (int):           training epoch
        save_dir (str):        directory to save plots
    """
    # Compute & Plot DTW alignment of time series
    alignment = dtw(v_extrap, omni_ts, keep_internals=True)

    ax = alignment.plot(type="twoway")
    ax.set_title(f'Sample {sample_id} - Epoch {epoch}: DTW score = {alignment.normalizedDistance:.4f}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed (km/s)')
    ax.set_ylim(200, 800)
    # label the plotted lines with legend
    lines = ax.get_lines()
    lines[0].set_label('NN HUX')
    lines[1].set_label('OMNI')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_dir, f'dtw_s{sample_id}_ep{epoch}.png'), dpi=150)
    plt.close()
