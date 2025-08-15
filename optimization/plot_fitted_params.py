# Python script to print the 5 parameter values distribution across CRs

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the CSV file into a DataFrame
df_params = pd.read_csv('results/fitted_wsa_params.csv')

# Load the metrics data to get PCC values
df_metrics = pd.read_csv('results/fitted_wsa_metrics.csv')

# Merge the dataframes on CR column to get parameters and PCC in the same dataframe
df = pd.merge(df_params, df_metrics[['CR', 'PCC']], on='CR')

# Classify PCC values into categories
df['PCC_class'] = pd.cut(df['PCC'], 
                         bins=[-1, 0.3, 0.7, 1], 
                         labels=['Bad', 'Fair', 'Good'], 
                         include_lowest=True)

# Create a figure with 7 subplots (7 rows, 1 column)
fig, axs = plt.subplots(7, 1, figsize=(7, 12), sharex=True)
axs = axs.flatten()

# Create color map based on PCC classification
color_map = {'Bad': 'red', 'Fair': 'orange', 'Good': 'green'}
df['color'] = df['PCC_class'].map(color_map)

# List of parameters to plot
parameters = ['vmin', 'vmax', 'alpha', 'beta', 'a1', 'w', 'a2']

# Plot each parameter wrt CR in a separate subplot
for i, param in enumerate(parameters):
    ax = axs[i]  # Use the correct subplot
    # Create scatter plots for each PCC category
    for category in ['Bad', 'Fair', 'Good']:
        mask = df['PCC_class'] == category
        ax.scatter(df.loc[mask, 'CR'], df.loc[mask, param], s=30,
                  alpha=0.7, color=color_map[category], 
                  label=category if param == 'vmin' else None,
                  edgecolor='black')
    
    # Add a horizontal line for the default value
    if param in ['vmin', 'vmax', 'alpha', 'beta', 'a1', 'w', 'a2']:
        default_values = {
            'vmin': 250.0,
            'vmax': 750.0,
            'alpha': 0.222,
            'beta': 1.25,
            'a1': 0.80,
            'w': 0.028,
            'a2': 3.0
        }
        ax.axhline(y=default_values[param], color='blue', linestyle='--',
                   label='Default' if param == 'vmin' else None, alpha=0.3)
    
    # Only add X-label to the last subplot
    if i == len(parameters) - 1:  # Last subplot
        ax.set_xlabel('CR', fontsize=12)
    else:
        ax.set_xlabel('')  # Remove X-label for other subplots

    ax.set_ylabel(param, fontsize=15)
    ax.minorticks_on()  # Enable minor ticks
    ax.tick_params(axis='both', which='major', length=5, labelsize=12)
    ax.tick_params(axis='both', which='minor', length=3, labelsize=10)
    ax.grid(False)
    
    # Only add legend to the first subplot
    if param == 'vmin':
        handles = [
            mpatches.Patch(color='red', label='PCC < 0.3'),
            mpatches.Patch(color='orange', label='0.3 ≤ PCC ≤ 0.7'),
            mpatches.Patch(color='green', label='PCC > 0.7'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Default Value')
        ]
        ax.legend(handles=handles, ncol=2, fontsize=13, bbox_to_anchor=(0.5, 1.7), loc='upper center')

# Align all y-axis labels
fig.align_ylabels(axs)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 1])  # Make room for suptitle

# Save the figure
plt.savefig('results/parameter_distributions.png')
plt.close()