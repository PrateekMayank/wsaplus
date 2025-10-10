import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import sunpy
from sunpy.time import TimeRange
from sunpy.net import Fido, attrs as a
from sunpy import timeseries as ts
from sunpy.coordinates.sun import carrington_rotation_time
import warnings
warnings.filterwarnings('ignore')  # Suppress SunPy warnings
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 1.2
#-----------------------------------------------------------
def get_sunspot_number_for_cr(cr_number):
    """
    Get average sunspot number for a given Carrington Rotation number.
    
    Parameters:
    -----------
    cr_number : int
        Carrington Rotation number
        
    Returns:
    --------
    float
        Average sunspot number for that CR period, or NaN if data not available
    """
    
    # Get start and end times for the CR
    cr_start = carrington_rotation_time(cr_number)
    cr_end = carrington_rotation_time(cr_number + 1)
    time_range = TimeRange(cr_start, cr_end)
    
    # Search for NOAA indices data
    result = Fido.search(a.Time(time_range), a.Instrument('noaa-indices'))
        
    # Fetch the data
    f_noaa_indices = Fido.fetch(result)
    
    # Create TimeSeries and truncate to our time range
    noaa = ts.TimeSeries(f_noaa_indices, source='noaaindices').truncate(time_range)

    sunspot_col = 'sunspot RI'
        
    # Calculate average sunspot number for this CR
    avg_sunspot = noaa.data[sunspot_col].mean()
    print(avg_sunspot)
    return float(avg_sunspot)
        

def get_sunspot_numbers_for_crs(cr_list):
    """
    Get sunspot numbers for a list of CR numbers with caching.
    
    Parameters:
    -----------
    cr_list : list
        List of Carrington Rotation numbers
        
    Returns:
    --------
    dict
        Dictionary mapping CR numbers to sunspot numbers
    """
    # Check if we have cached data
    cache_file = 'CR_metrics/sunspot_cache.csv'
    
    if os.path.exists(cache_file):
        print("Loading cached sunspot data...")
        cache_df = pd.read_csv(cache_file)
        sunspot_dict = dict(zip(cache_df['cr'], cache_df['sunspot_number']))
    else:
        sunspot_dict = {}
    
    # Get missing data
    missing_crs = [cr for cr in cr_list if cr not in sunspot_dict]
    
    if missing_crs:
        print(f"Fetching sunspot data for {len(missing_crs)} CRs...")
        for i, cr in enumerate(missing_crs):
            print(f"Processing CR {cr} ({i+1}/{len(missing_crs)})")
            sunspot_dict[cr] = get_sunspot_number_for_cr(cr)
        
        # Save updated cache
        cache_data = pd.DataFrame(list(sunspot_dict.items()), 
                                columns=['cr', 'sunspot_number'])
        cache_data.to_csv(cache_file, index=False)
        print(f"Cached sunspot data saved to {cache_file}")
    
    return sunspot_dict
#-----------------------------------------------------------

# Path to metrics CSV file
csv_path = os.path.join('CR_plots', 'metrics.csv')
metrics_df = pd.read_csv(csv_path)

# Sort by CR number
metrics_df = metrics_df.sort_values('cr')

# Create output directory if it doesn't exist
os.makedirs('CR_metrics', exist_ok=True)

# Normlaizing the metrices wrt maximum value
MAE_MAX = metrics_df['mae_wsa_def'].max()  # worst‐case RMSE
RMSE_MAX  = metrics_df['rmse_wsa_def'].max()  # worst‐case MAE
DTW_MAX  = metrics_df['dtw_wsa_def'].max()  # worst‐case DTW
DTW_MAX  = 100.0    # worst‐case DTW
PCC_MIN, PCC_MAX = -1.0, 1.0  # worst/best PCC

mae_norm = 1 - (metrics_df['mae_wsa_def'] / MAE_MAX)
cc_norm = (metrics_df['cc_wsa_def'] - PCC_MIN) / (PCC_MAX - PCC_MIN)
rmse_norm = 1 - (metrics_df['rmse_wsa_def'] / RMSE_MAX)
dtw_norm = 1 - (metrics_df['dtw_wsa_def'] / DTW_MAX)

mae_pred_norm = 1 - (metrics_df['mae_pred'] / MAE_MAX)
cc_pred_norm = (metrics_df['cc_pred'] - PCC_MIN) / (PCC_MAX - PCC_MIN)
rmse_pred_norm = 1 - (metrics_df['rmse_pred'] / RMSE_MAX)
dtw_pred_norm = 1 - (metrics_df['dtw_pred'] / DTW_MAX)

# Combine into a single fit score (equal weights)
metrics_df['fit_score'] = (mae_pred_norm + cc_pred_norm + rmse_pred_norm + dtw_pred_norm) / 4
metrics_df['fit_score_def'] = (mae_norm + cc_norm + rmse_norm + dtw_norm) / 4

# Define metric pairs to plot together
metric_pairs = [
    ('rmse_wsa_def', 'rmse_pred'),
    ('mae_wsa_def', 'mae_pred'),
    ('dtw_wsa_def', 'dtw_pred'),
    ('cc_wsa_def', 'cc_pred')
]

# Custom ylimits for the y-axis
y_limits = {
    'rmse_wsa_def': (0, 270),
    'rmse_pred': (0, 270),
    'cc_wsa_def': (-0.7, 1.5),
    'cc_pred': (-0.7, 1.5),
    'mae_wsa_def': (0, 240),
    'mae_pred': (0, 240),
    'dtw_wsa_def': (0, 110),
    'dtw_pred': (0, 110),
    'fit_score': (0, 1.0)
}

#-----------------------------------------------------------
# Define the manifest and splits
manifest = metrics_df['cr'].tolist()  # Assuming 'cr' column corresponds to the manifest
val_manifest = [manifest[i] for i in range(len(manifest)) if i % 6 == 0]
test_manifest = [manifest[i] for i in range(len(manifest)) if i % 6 == 1]
train_manifest = [manifest[i] for i in range(len(manifest)) if i % 6 != 0 and i % 6 != 1]

# Add a column to indicate the dataset split
def get_split(cr):
    if cr in val_manifest:
        return 'Validation'
    elif cr in test_manifest:
        return 'Test'
    else:
        return 'Train'

metrics_df['split'] = metrics_df['cr'].apply(get_split)

# Define colors for each split
split_colors = {'Train': 'C0', 'Validation': 'purple', 'Test': 'C1'}
#-----------------------------------------------------------

# Create a figure with 9 subplots (3 rows, 3 columns)
fig, axs = plt.subplots(3, 3, figsize=(20, 12), gridspec_kw={'height_ratios': [1, 1, 0.8]})
axs = axs.flatten()


# Plot each pair wrt CR in the same subplot, with thicker vertical "bars" connecting the two values
for i, (m1, m2) in enumerate(metric_pairs):
    ax = axs[i]
    
    # vertical lines (bars) connecting m1 and m2 at each CR
    for x, y1, y2 in zip(metrics_df['cr'], metrics_df[m1], metrics_df[m2]):
        diff = y2 - y1
        if m1 in ('cc_wsa_def'):
            # For other metrics, lower is better
            color = 'red' if diff < 0 else 'green'
            alpha = 0.05 if diff < 0 else 0.08
            
        else:
            # For correlation, higher is better
            color = 'red' if diff > 0 else 'green'
            alpha = 0.08 if diff > 0 else 0.08       

        ax.vlines(x, y1, y2, color=color, alpha=alpha, linewidth=4)
    
    # Fit polynomial trendlines
    x = metrics_df['cr'].values
    y1 = metrics_df[m1].values
    y2 = metrics_df[m2].values
    
    # polynomial degree
    deg = 5
    
    # fit polynomials
    coef1 = np.polyfit(x, y1, deg)
    coef2 = np.polyfit(x, y2, deg)
    
    # generate smooth x
    x_smooth = np.linspace(x.min(), x.max(), 200)
    
    # evaluate polynomials
    y1_smooth = np.polyval(coef1, x_smooth)
    y2_smooth = np.polyval(coef2, x_smooth)
    
    # plot curves with higher alpha
    ax.plot(x_smooth, y1_smooth, color='grey', linestyle='-', linewidth=1.5, alpha=0.45)
    ax.plot(x_smooth, y2_smooth, color='C0', linestyle='-', linewidth=1.5, alpha=0.45)
    
    # scatter plots for each metric of WSA default
    ax.scatter(metrics_df['cr'], metrics_df[m1], alpha=0.7, color='grey', label='WSA Default')
    # scatter plots for each metric of WSA+ with color coding based on the split
    for split, color in split_colors.items():
        split_mask = metrics_df['split'] == split
        ax.scatter(
            metrics_df.loc[split_mask, 'cr'], 
            metrics_df.loc[split_mask, m2], 
            alpha=0.7, 
            facecolor=color,  # Distinct face color for each split
            edgecolor='C0',   # Consistent edge color for all points
            linewidth=2,    # Edge line width
            s=30,           # Marker size
            label=f'WSA+ ({split})'
        )

    # Add legend
    ax.legend(ncol=2, loc='upper center', fontsize=12)
    
    # compute means and horizontal lines
    m1_mean = metrics_df[m1].mean()
    m2_mean = metrics_df[m2].mean()
    ax.axhline(m1_mean, color='grey', linestyle='--')
    ax.axhline(m2_mean, color='C0', linestyle='--')
    
    # Format title based on metric name
    title_map = {
        'rmse_wsa_def': 'RMSE', 
        'cc_wsa_def': 'PCC',
        'mae_wsa_def': 'MAE',
        'dtw_wsa_def': 'DTW'
    }
    
    # primary axis settings
    subplot_labels = ['(a1)', '(a2)', '(a3)', '(a4)']
    metrics = [pair[0] for pair in metric_pairs]
    metric_to_index = {metric: i for i, metric in enumerate(metrics)}
    ax.set_title(f'{subplot_labels[metric_to_index[m1]]} {title_map[m1]} vs CR', fontsize=16)
    ax.set_xlabel('CR', fontsize=14)
    ax.set_ylabel(f'{title_map[m1]} Value', fontsize=14)
    
    # Set y-limits from the dictionary
    if m1 in y_limits:
        ax.set_ylim(y_limits[m1][0], y_limits[m1][1])
    
    # Set exactly 5 ticks on y-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    
    ax.tick_params(direction='inout', axis='both', which='major', right='True', length=6, width=1, labelsize=12)
    ax.tick_params(direction='inout', axis='both', which='minor', right='True', length=4, width=0.6, labelsize=12)
    ax.minorticks_on()

    # Create legend
    ax.legend(ncol=2, loc='upper center', fontsize=12)
    ax.grid(False)

# 5th subplot: Fit Score
ax_fit = axs[4]
x = metrics_df['cr'].values
y = metrics_df['fit_score'].values

# Get sunspot numbers for CRs
sunspot_dict = get_sunspot_numbers_for_crs(metrics_df['cr'].tolist())
# Add sunspot numbers to your dataframe
metrics_df['sunspot_number'] = metrics_df['cr'].map(sunspot_dict)

# Fit polynomial for trendline
coef = np.polyfit(x, y, deg=5)
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = np.polyval(coef, x_smooth)

# Add mean line
fit_score_mean = metrics_df['fit_score'].mean()
ax_fit.plot(x_smooth, y_smooth, color='grey', linestyle='-', linewidth=2, alpha=0.3)
ax_fit.axhline(fit_score_mean, color='black', linestyle='--', alpha=0.6, label=f'Mean')
# Set exactly 5 ticks on y-axis
ax_fit.yaxis.set_major_locator(ticker.MaxNLocator(5))

# Color points based on score classification
metrics_df['score_class'] = pd.cut(metrics_df['fit_score'], 
                         bins=[0, 0.6, 0.75, 1], 
                         labels=['Bad', 'Fair', 'Good'], 
                         include_lowest=True)

color_map = {'Bad': 'red', 'Fair': 'orange', 'Good': 'green'}
for category in ['Bad', 'Fair', 'Good']:
    mask = metrics_df['score_class'] == category
    if sum(mask) > 0:  # Only plot if there are points in this category
        ax_fit.scatter(metrics_df.loc[mask, 'cr'], metrics_df.loc[mask, 'fit_score'], 
                      alpha=0.9, color=color_map[category], 
                      label=category, s=50)

ax_fit.set_title('(b) Fit Score vs CR', fontsize=16)
ax_fit.set_xlabel('CR', fontsize=14)
ax_fit.set_ylabel('Fit Score', fontsize=14)
ax_fit.set_ylim(0.3, 1.07)

# Create secondary y-axis for sunspot number
ax_sunspot = ax_fit.twinx()

# Plot sunspot data on secondary axis (only for CRs with valid data)
valid_data = metrics_df.dropna(subset=['sunspot_number'])
if len(valid_data) > 0:
    ax_sunspot.fill_between(valid_data['cr'], valid_data['sunspot_number'], 
                       color='teal', alpha=0.3, 
                       label='Sunspot Number')
    ax_sunspot.plot(valid_data['cr'], valid_data['sunspot_number'], 
                   color='teal', alpha=0.3, linewidth=1, 
                   label='Sunspot Number', marker='o', markersize=3)
    ax_sunspot.set_ylabel('Sunspot Number', fontsize=14, color='teal', labelpad=-15)
    ax_sunspot.tick_params(axis='y', labelcolor='teal', labelsize=12)
    ax_sunspot.set_ylim(0, valid_data['sunspot_number'].max() * 4)  # Set y-limits based on data
    ax_sunspot.text(0.33, 0.02, 'SC24', transform=ax_sunspot.transAxes, fontsize=14, color='teal')
    ax_sunspot.text(0.85, 0.02, 'SC25', transform=ax_sunspot.transAxes, fontsize=14, color='teal')
    ax_sunspot.set_yticks([0, 50, 100])

ax_fit.tick_params(direction='inout', axis='both', which='major', length=6, width=1, labelsize=12)
ax_fit.tick_params(direction='inout', axis='both', which='minor', length=4, width=0.6, labelsize=12)
ax_fit.minorticks_on()
ax_fit.legend(ncol=4, loc ='upper center', fontsize=12)
ax_fit.grid(False)

# 6th subplot: Summary Statistics
ax_summary = axs[5]
ax_summary.axis('off')  # Turn off axis for text display

# Compute summary statistics
avg_score_wsa = metrics_df['fit_score_def'].mean()
avg_score_pred = metrics_df['fit_score'].mean()
improvement_score = -1 * (avg_score_wsa - avg_score_pred) / avg_score_wsa * 100

avg_rmse_wsa = metrics_df['rmse_wsa_def'].mean()
avg_rmse_pred = metrics_df['rmse_pred'].mean()
improvement_rmse = (avg_rmse_wsa - avg_rmse_pred) / avg_rmse_wsa * 100

avg_cc_wsa = metrics_df['cc_wsa_def'].mean()
avg_cc_pred = metrics_df['cc_pred'].mean()
improvement_cc = (avg_cc_pred - avg_cc_wsa) / abs(avg_cc_wsa) * 100

avg_mae_wsa = metrics_df['mae_wsa_def'].mean()
avg_mae_pred = metrics_df['mae_pred'].mean()
improvement_mae = (avg_mae_wsa - avg_mae_pred) / avg_mae_wsa * 100

avg_dtw_wsa = metrics_df['dtw_wsa_def'].mean()
avg_dtw_pred = metrics_df['dtw_pred'].mean()
improvement_dtw = (avg_dtw_wsa - avg_dtw_pred) / avg_dtw_wsa * 100

avg_score_wsa = metrics_df['fit_score_def'].mean()
avg_score_pred = metrics_df['fit_score'].mean()
improvement_dtw = (avg_dtw_wsa - avg_dtw_pred) / avg_dtw_wsa * 100

# Count score classifications
score_counts = metrics_df['score_class'].value_counts().sort_index()
total_crs = len(metrics_df)

summary_text = (
    f"\nPerformance Summary of 129 CRs\n"
    f"{'='*45}\n\n"
    f"{'Avg. Metric':<8}{'WSA':>10}{'WSA+':>10}{'Improvement':>15}\n"
    f"{'-'*45}\n"
    f"\n{'Fit Score':<12}{avg_score_wsa:10.2f}{avg_score_pred:10.2f}{improvement_score:9.1f}%\n"
    f"{'DTW':<12}{avg_dtw_wsa:10.2f}{avg_dtw_pred:10.2f}{improvement_dtw:9.1f}%\n"
    f"{'RMSE':<12}{avg_rmse_wsa:10.2f}{avg_rmse_pred:10.2f}{improvement_rmse:9.1f}%\n"
    f"{'MAE':<12}{avg_mae_wsa:10.2f}{avg_mae_pred:10.2f}{improvement_mae:9.1f}%\n"
    f"{'PCC':<12}{avg_cc_wsa:10.2f}{avg_cc_pred:10.2f}{improvement_cc:9.1f}%\n")
#     f"Fit Score Classification of {total_crs:>3} CRs\n"
#     f"{'-'*25}\n"
#     f"Good:  {score_counts.get('Good', 0):>6} CRs ({score_counts.get('Good', 0)/total_crs*100:4.1f}%)\n"
#     f"Fair:  {score_counts.get('Fair', 0):>6} CRs ({score_counts.get('Fair', 0)/total_crs*100:4.1f}%)\n"
#     f"Bad:   {score_counts.get('Bad', 0):>6} CRs ({score_counts.get('Bad', 0)/total_crs*100:4.1f}%)\n\n"
# )

ax_summary.text(0.05, 0.9, summary_text, transform=ax_summary.transAxes, 
                fontsize=13, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

# 7th subplot: DTW Distribution
bins = 15
ax_rmse_dist = axs[6]
metrics_df[['dtw_wsa_def', 'dtw_pred']].plot(kind='hist', bins=bins, alpha=0.5, ax=ax_rmse_dist)
ax_rmse_dist.set_title('(c1) DTW Distribution', fontsize=16)
ax_rmse_dist.set_xlabel('DTW Value', fontsize=14)
ax_rmse_dist.set_ylabel('Frequency', fontsize=14)
ax_rmse_dist.legend(['WSA', 'WSA+'], ncol=2, fontsize=12)
ax_rmse_dist.tick_params(direction='inout', axis='both', which='major', right='True', length=6, width=1, labelsize=12)
ax_rmse_dist.tick_params(direction='inout', axis='both', which='minor', right='True', length=4, width=0.6, labelsize=12)
ax_rmse_dist.minorticks_on()

# 8th subplot: RMSE Distribution
ax_rmse_dist = axs[7]
metrics_df[['rmse_wsa_def', 'rmse_pred']].plot(kind='hist', bins=bins, alpha=0.5, ax=ax_rmse_dist)
ax_rmse_dist.set_title('(c2) RMSE Distribution', fontsize=16)
ax_rmse_dist.set_xlabel('RMSE Value', fontsize=14)
ax_rmse_dist.set_ylabel('Frequency', fontsize=14)
ax_rmse_dist.legend(['WSA', 'WSA+'], ncol=2, fontsize=12)
ax_rmse_dist.tick_params(direction='inout', axis='both', which='major', right='True', length=6, width=1, labelsize=12)
ax_rmse_dist.tick_params(direction='inout', axis='both', which='minor', right='True', length=4, width=0.6, labelsize=12)
ax_rmse_dist.minorticks_on()

# 9th subplot: CC Distribution
ax_cc_dist = axs[8]
metrics_df[['fit_score_def', 'fit_score']].plot(kind='hist', bins=bins, alpha=0.5, ax=ax_cc_dist)
ax_cc_dist.set_title('(c3) Fit Score Distribution', fontsize=16)
ax_cc_dist.set_xlabel('Fit Score Value', fontsize=14)
ax_cc_dist.set_ylabel('Frequency', fontsize=14)
ax_cc_dist.legend(['WSA', 'WSA+'], ncol=2, fontsize=12)
ax_cc_dist.tick_params(direction='inout', axis='both', which='major', right='True', length=6, width=1, labelsize=12)
ax_cc_dist.tick_params(direction='inout', axis='both', which='minor', right='True', length=4, width=0.6, labelsize=12)
ax_cc_dist.minorticks_on()

# Align labels
fig.align_ylabels(axs[:9])

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
plt.savefig('CR_metrics/metrics_distribution_new.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the classification data
metrics_df[['cr', 'fit_score', 'score_class']].to_csv('CR_metrics/fit_scores.csv', index=False)


# Print summary information
print(f"Average RMSE improvement: {improvement_rmse:.1f}%")
print(f"Average PCC improvement: {improvement_cc:.1f}%")
print(f"Average MAE improvement: {improvement_mae:.1f}%")
print(f"Average DTW improvement: {improvement_dtw:.1f}%")