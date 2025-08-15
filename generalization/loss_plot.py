import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the log
df = pd.read_csv('train_log.csv', delimiter='\t')

# Ensure we have epochs for x-axis
if 'epoch' in df.columns:
    x, xlabel = df['epoch'], 'Epoch'
else:
    x, xlabel = df.index, 'Iteration'

# Create a figure for losses
plt.figure(figsize=(8, 5))

# Plot all loss/metric columns
metrics = ['train_map', 'val_map']

# Normalize each metric by its maximum value
normalized_metrics = {}
for metric in metrics:
    max_value = df['train_map'].max()
    normalized_metrics[metric] = df[metric] / max_value

colors = ['C0', 'C2']
metric_labels = ['Training Loss', 'Validation Loss']
for metric in metrics:
    plt.plot(x, normalized_metrics[metric], label=metric_labels[metrics.index(metric)], 
             alpha=0.9, linewidth=3, color=colors[metrics.index(metric)])

plt.xlabel(xlabel, fontsize=14, labelpad=10)
plt.ylabel('Normalized Loss Values', fontsize=14, labelpad=10)
plt.minorticks_on()
plt.tick_params(axis='both', which='major', length=5, labelsize=12)
plt.tick_params(axis='both', which='minor', length=3, labelsize=10)
plt.ylim([0.12, 0.4])  # Set y-limits to accommodate normalized values
plt.grid(False)
plt.legend(ncol=2, fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 0.99))
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)