# WSA+ Model: Training and Plotting Pipeline

This repository contains scripts for training, evaluating, and visualizing the **WSA+** model — a neural enhancement of the traditional Wang–Sheeley–Arge (WSA) solar wind relation.  
The workflow is divided into two main stages:

1. **Optimization** — per Carrington Rotation (CR) fitting of WSA empirical parameters using in-situ solar wind observations.
2. **Generalization** — training a neural network to predict optimized WSA speed maps directly from magnetogram-derived features.

---

## End-to-End Workflow

flowchart TD
  A["Input Synoptic Magnetograms (GONG/HMI)"] --> B["Optimization: fit_wsa_params.py"]
  B --> C["Parameter Plots: plot_fitted_params.py"]
  C --> D["Generalization Training: train.py"]
  D --> E["Loss Curves: loss_plot.py"]
  D --> F["CR-wise Visuals: post_training_CR_plots.py"]
  D --> G["CR-wise Metrics: post_training_CR_metrices.py"]
  D --> H["2D Speed Maps: 3_2D_map_plot.py"]
  D --> I["In-situ Panels: in-situ_maps.py"]
  D --> J["Dataset Comparison: dataset_comparision.py"]
  A -. "optional" .-> D
