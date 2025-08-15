import os
import time
import random
import logging
import csv
import torch.nn.functional as F

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import (
    DATA_MANIFEST, DEVICE,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    LR, EPOCHS, WEIGHT_DECAY,
    SEED, LOG_FILE, CHECKPOINT_DIR
)
from dataset import CarringtonDataset
from model import WSASurrogateModel
from loss import MapLoss, WSAResidualLoss
import visualize

# For deterministic cublas kernels (PyTorch recommendation)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

# Add anomaly detection - will help trace NaN issues in backprop
#torch.autograd.set_detect_anomaly(True)

    
def main():

    # ---------- Logging Configuration ----------
    # Only print the message
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # ---------- Sobel Kernels ----------
    # Sobel kernels for edge detection
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=DEVICE).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)

    # ---------- Reproducibility ----------
    #Ensuring Reproducibility by eliminating randomness as a variable
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Avoid TF32 variability
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    def seed_worker(worker_id):
        worker_seed = (torch.initial_seed() % 2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    # ---------- Train/Val Split ----------
    # Split manifest dataset into train/val
    manifest = DATA_MANIFEST.copy()

    # Split into train (85), validation (22), and test (22) sets
    val_manifest = [manifest[i] for i in range(len(manifest)) if i % 6 == 0]
    test_manifest = [manifest[i] for i in range(len(manifest)) if i % 6 == 1] 
    train_manifest = [manifest[i] for i in range(len(manifest)) if i % 6 != 0 and i % 6 != 1]

    logging.info(f"Train/Val/Test split: {len(train_manifest)}/{len(val_manifest)}/{len(test_manifest)} rotations")

    # ---------- DataLoaders ----------
    # Datasets and DataLoaders: training and validation
    train_ds = CarringtonDataset(manifest=train_manifest)
    val_ds = CarringtonDataset(manifest=val_manifest)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False
    )

    # ---------- Model & Losses ----------
    model = WSASurrogateModel().to(DEVICE)
    # 1a. Freeze encoder for the first N epochs (e.g. 10)
    for p in model.backbone.parameters():
        p.requires_grad = False
    
    map_loss = MapLoss().to(DEVICE)
    res_loss = WSAResidualLoss().to(DEVICE)

    # ---------- Optimizer & Scheduler ----------
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW([
        {'params': no_decay,  'weight_decay': 0.0},
        {'params': decay,     'weight_decay': WEIGHT_DECAY},
    ], lr=LR)
    
    # switch to ReduceLROnPlateau:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    mode='min', factor=0.9, patience=10, min_lr=1e-6)

    # ---------- Logging CSV ----------
    os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
    csv_file = open(LOG_FILE, 'w', newline='')
    fieldnames = ['epoch', 'lr', 'train_map', 'val_map', 'time']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    #********************************************************************
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------- Training Loop ---------
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_map_sum = 0.0
        train_batches = 0

        # 1b. Unfreeze after warmup epochs
        if epoch == 11:
            for p in model.backbone.parameters():
                p.requires_grad = True

        for batch in train_loader:
            x_map = batch['map'].to(DEVICE, non_blocking=True)          # [B, 2/3, H, W]
            wsa_map = batch['wsap'].to(DEVICE, non_blocking=True)       # [B, H, W]
            lat_idx = batch['lat_idx'].to(DEVICE, non_blocking=True)    # [B, T]
            omni_ts = batch['omni'].to(DEVICE, non_blocking=True)       # [B, T]
            omega   = batch['omega'].to(DEVICE)                         # scalar [B], ignore non_blocking
            cr = batch['cr'].to(DEVICE)  # Carrington Rotation number
            
            if torch.isnan(x_map).any() or torch.isinf(x_map).any():
                raise RuntimeError("Found NaN in input map!")

            optimizer.zero_grad()

            # Forward pass    
            pred_map = model(x_map)
            l_map = map_loss(pred_map, wsa_map)

            loss = l_map
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Debugging for NaN in loss
            if torch.isnan(pred_map).any():
                print("Found NaN in Model Prediction during Training!")

            train_map_sum += l_map.item()
            
            train_batches += 1

        #*********************************************************************

        # Validation
        model.eval()
        val_map_sum = 0.0

        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x_map = batch['map'].to(DEVICE)
                wsa_map = batch['wsap'].to(DEVICE)
                lat_idx = batch['lat_idx'].to(DEVICE)
                omni_ts = batch['omni'].to(DEVICE)
                omega   = batch['omega'].to(DEVICE)
                cr = batch['cr'].to(DEVICE)

                pred_map = model(x_map)
                l_map = map_loss(pred_map, wsa_map)

                val_map_sum += l_map.item()

                val_batches += 1

                # Debugging for NaN in validation loss
                if torch.isnan(pred_map).any():
                    print("Found NaN in Model Prediction during Validation!")

        train_map_avg = train_map_sum / train_batches
        val_map_avg = val_map_sum / val_batches

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        scheduler.step(val_map_avg)  # Step scheduler based on validation loss

        # Logging
        logging.info(
            f"Epoch {epoch}/{EPOCHS} "
            f"LR={current_lr:.2e} "
            f"TrainMap={train_map_avg:.4f} "
            f"ValMap={val_map_avg:.4f} "
            f"Time={epoch_time:.1f}s"
        )
        writer.writerow({
            'epoch': epoch,
            'lr': f"{current_lr:.2e}",
            'train_map': f"{train_map_avg:.4f}",
            'val_map': f"{val_map_avg:.4f}",
            'time': f"{epoch_time:.1f}"
        })
        csv_file.flush()

        # ------ Visualization ------
        # Plot 2D map comparisons every epoch
        VIS_SAMPLES = [1, 2, 3]         # sample indices for visualization
        for sid in VIS_SAMPLES:
            sample = val_ds[sid]
            x_s = sample['map'].unsqueeze(0).to(DEVICE)
            wsa_np = sample['wsap'].cpu().numpy()
            omni_np = sample['omni'].cpu().numpy()
            lat_idx_np = sample['lat_idx'].cpu().numpy()
            omega_val = sample['omega']

            with torch.no_grad():
                pred_np = model(x_s)[0, 0].cpu().numpy()
            visualize.plot_map_comparison(pred_np, wsa_np, sample_id=sid, epoch=epoch)

            # DTW plot every 5 epochs
            if epoch % 25 == 0:
                T = lat_idx_np.shape[0]
                v_in_np = pred_np[np.arange(T), lat_idx_np]
                v_in_t = torch.from_numpy(v_in_np.astype(np.float32)).unsqueeze(0).to(DEVICE)
                omega_t = torch.tensor([omega_val], dtype=torch.float32, device=DEVICE)
                #v_out_np = seq_loss.hux(v_in_t, omega_t)[0].cpu().numpy()
                #visualize.plot_dtw_alignment(v_out_np, omni_np, sample_id=sid, epoch=epoch)

        # ---------- Checkpointing ----------
        ckpt = {
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict()
        }
        
        # Save last epoch checkpoint
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, 'last.pt'))

    csv_file.close()

if __name__ == '__main__':
    main()
    
