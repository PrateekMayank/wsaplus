import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler

# Path to metrics CSV file
csv_path = os.path.join('CR_plots', 'metrics.csv')
metrics_df = pd.read_csv(csv_path)

# Sort by CR number
metrics_df = metrics_df.sort_values('cr')

# Create output directory if it doesn't exist
os.makedirs('CR_metrics', exist_ok=True)

# Split data into train/val/test using the same logic as in train.py
crs = metrics_df['cr'].unique()
val_crs = [crs[i] for i in range(len(crs)) if i % 6 == 0]
test_crs = [crs[i] for i in range(len(crs)) if i % 6 == 1]
train_crs = [cr for cr in crs if cr not in val_crs and cr not in test_crs]

# Create subsets based on the splits
train_df = metrics_df[metrics_df['cr'].isin(train_crs)]
val_df = metrics_df[metrics_df['cr'].isin(val_crs)]
test_df = metrics_df[metrics_df['cr'].isin(test_crs)]

# Calculate fit scores for all datasets using MinMaxScaler
# Define a function to calculate fit scores
def calculate_fit_score(df):
    # Normalize metrics (lower is better for rmse, mae, dtw; higher is better for cc)
    mae_norm = 1 - (df['mae_pred'] / df['mae_wsa_def'].max())
    cc_norm = (df['cc_pred'] - (-1.0)) / (1.0 - (-1.0))  # Scale from [-1,1] to [0,1]
    rmse_norm = 1 - (df['rmse_pred'] / df['rmse_wsa_def'].max())
    dtw_norm = 1 - (df['dtw_pred'] / 100.0)  # Using 100 as max DTW
    
    # Calculate WSA fit scores
    mae_wsa_norm = 1 - (df['mae_wsa_def'] / df['mae_wsa_def'].max())
    cc_wsa_norm = (df['cc_wsa_def'] - (-1.0)) / (1.0 - (-1.0))
    rmse_wsa_norm = 1 - (df['rmse_wsa_def'] / df['rmse_wsa_def'].max())
    dtw_wsa_norm = 1 - (df['dtw_wsa_def'] / 100.0)
    
    # Combine into single scores
    df['fit_score'] = (mae_norm + cc_norm + rmse_norm + dtw_norm) / 4
    df['fit_score_wsa'] = (mae_wsa_norm + cc_wsa_norm + rmse_wsa_norm + dtw_wsa_norm) / 4
    
    return df

# Calculate fit scores for each dataset
train_df = calculate_fit_score(train_df)
val_df = calculate_fit_score(val_df)
test_df = calculate_fit_score(test_df)

# Calculate average metrics for each dataset
metrics = {
    'mae': {'label': '(a) MAE', 'lower_better': True},
    'rmse': {'label': '(b) RMSE', 'lower_better': True},
    'cc': {'label': '(c) PCC', 'lower_better': False},
    'dtw': {'label': '(d) DTW', 'lower_better': True},
    'fit_score': {'label': '(e) Fit Score', 'lower_better': False}
}

# Calculate averages
datasets = {
    'train': {'df': train_df, 'label': 'Training'},
    'val': {'df': val_df, 'label': 'Validation'},
    'test': {'df': test_df, 'label': 'Test'}
}

# Store results
results = {}
for metric_key, metric_info in metrics.items():
    results[metric_key] = {'wsa': {}, 'nn': {}}
    for ds_key, ds_info in datasets.items():
        df = ds_info['df']
        if metric_key == 'fit_score':
            results[metric_key]['wsa'][ds_key] = round(df['fit_score_wsa'].mean(), 3)
            results[metric_key]['nn'][ds_key] = round(df['fit_score'].mean(), 3)
        else:
            wsa_col = f"{metric_key}_wsa_def"
            nn_col = f"{metric_key}_pred"
            results[metric_key]['wsa'][ds_key] = round(df[wsa_col].mean(), 2)
            results[metric_key]['nn'][ds_key] = round(df[nn_col].mean(), 2)

# Create visualization
fig, axs = plt.subplots(2, 3, figsize=(18, 8))
axs = axs.flatten()

# Define custom y-limits for each metric
y_limits = {
    'mae': (0, 160),
    'rmse': (0, 200),
    'cc': (0.0, 0.8),
    'dtw': (0, 70),
    'fit_score': (0.0, 1.0)
}

# Plot metrics
for i, (metric_key, metric_info) in enumerate(metrics.items()):
        
    ax = axs[i]
    
    # Set up data for plotting
    x_labels = ['Training', 'Validation', 'Test']
    x = np.arange(len(x_labels))
    width = 0.35
    
    wsa_values = [results[metric_key]['wsa']['train'], 
                  results[metric_key]['wsa']['val'], 
                  results[metric_key]['wsa']['test']]
    nn_values = [results[metric_key]['nn']['train'], 
                 results[metric_key]['nn']['val'], 
                 results[metric_key]['nn']['test']]
    
    # Create bars
    bars1 = ax.bar(x - width/2, wsa_values, width, label='WSA', color='C0', alpha=0.8)
    bars2 = ax.bar(x + width/2, nn_values, width, label='WSA+', color='grey', alpha=0.8)
    
    # Add labels and formatting
    ax.set_xlabel('Dataset', fontsize=16, labelpad=10)
    ax.set_ylabel(f'{metric_info["label"]}', fontsize=16, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(ncol=2, loc='upper center', fontsize=14)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=6, labelsize=12)
    ax.tick_params(axis='both', which='minor', length=4, labelsize=10)

    # Set custom y-axis limits
    if metric_key in y_limits:
        ax.set_ylim(y_limits[metric_key])
    
    # Set number of y-axis ticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# 6th subplot: Summary text
ax_summary = axs[5]
ax_summary.axis('off')  # Turn off axis for text display

# Create summary text
summary_text = (
    f"Dataset Performance Summary\n"
    f"{'='*30}\n\n"
    f"{'Dataset':<12}{'Train':>8}{'Val':>8}{'Test':>8}\n"
    f"{'-'*36}\n"
)

for metric_key, metric_info in metrics.items():
    summary_text += (
        f"{metric_info['label']:<12}"
        f"{results[metric_key]['nn']['train']:>8}"
        f"{results[metric_key]['nn']['val']:>8}"
        f"{results[metric_key]['nn']['test']:>8}\n"
    )

ax_summary.text(0.05, 0.8, summary_text, transform=ax_summary.transAxes, 
                fontsize=12, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

# Add width space and adjust layout
plt.subplots_adjust(wspace=2)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
plt.savefig('CR_metrics/dataset_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Dataset comparison plot created: CR_metrics/dataset_comparison.png")