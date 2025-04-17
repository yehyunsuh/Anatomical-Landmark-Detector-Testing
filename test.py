"""
test.py

Testing utilities for anatomical landmark detection using U-Net.

This module contains functions to test the trained model with or without
ground truth labels, and to optionally visualize and save predicted results.

Author: Yehyun Suh  
Date: 2025-04-15  
"""

import os
import torch
import pandas as pd
from tqdm import tqdm

from data_loader import dataloader
from visualization import overlay_pred_masks_w_label, overlay_pred_masks_wo_label


def test_model_w_label(args, model, device, test_loader):
    """
    Evaluate model on test set with available ground truth labels.
    Computes pixel-wise landmark errors and saves them as CSV.

    Args:
        args (Namespace): Configuration arguments.
        model (nn.Module): Trained model.
        device (str): Device ('cuda' or 'cpu').
        test_loader (DataLoader): Test set dataloader.
    """
    model.eval()
    all_pred_coords = []
    all_gt_coords = []
    image_names = []

    with torch.no_grad():
        for idx, (images, image_name, landmarks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            outputs = model(images)

            # Predict coordinates
            probs = torch.sigmoid(outputs)
            B, C, H, W = probs.shape
            probs_flat = probs.view(B, C, -1)
            max_indices = probs_flat.argmax(dim=2)

            pred_coords = torch.zeros((B, C, 2), device=device)
            for b in range(B):
                for c in range(C):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, W)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            # Ground truth coordinates
            gt_coords = torch.tensor(landmarks, dtype=torch.float32, device=device)
            if gt_coords.ndim == 2:
                gt_coords = gt_coords.unsqueeze(0)

            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)
            image_names.append(image_name[0])

            overlay_pred_masks_w_label(image_name, images, pred_coords, gt_coords)

    # Stack results
    all_pred_coords = torch.cat(all_pred_coords, dim=0)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    dists = torch.norm(all_pred_coords - all_gt_coords, dim=2)  # [B, C]
    mean_dist = dists.mean().item()

    print(f"\nMean distance: {mean_dist:.4f}")
    for i in range(dists.shape[1]):
        print(f"Landmark {i+1}: {dists[:, i].mean().item():.4f}")

    # Compute stats
    dist_mean = dists.mean(dim=1).cpu().numpy()
    dist_std = dists.std(dim=1).cpu().numpy()
    dists_np = dists.cpu().numpy()
    n_landmarks = dists.shape[1]

    # image_names = [f"img_{i:03d}.png" for i in range(len(dists_np))]

    data = []
    for i in range(len(image_names)):
        row = [image_names[i], n_landmarks, dist_mean[i], dist_std[i]] + dists_np[i].tolist()
        data.append(row)

    columns = ["image_name", "n_landmarks", "dist_mean", "dist_std"]
    columns += [f"dist_landmark{j+1}" for j in range(n_landmarks)]
    df = pd.DataFrame(data, columns=columns)

    # Summary row
    overall_mean = dist_mean.mean()
    overall_std = dist_std.mean()
    overall_per_landmark = dists_np.mean(axis=0)

    summary_row = ["overall average", n_landmarks, overall_mean, overall_std] + overall_per_landmark.tolist()
    df.loc[len(df.index)] = summary_row

    # Save results
    os.makedirs("test_results", exist_ok=True)
    csv_path = os.path.join("test_results", f"{args.experiment_name}_landmark_distances.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved landmark distances to {csv_path}")

    # Save actual predicted coordinates
    pred_coords_np = all_pred_coords.cpu().numpy()
    image_sizes = [images.shape[-1], images.shape[-2]]  # W, H (assuming all images same size)

    pred_data = []
    for i in range(len(image_names)):
        row = [image_names[i], image_sizes[0], image_sizes[1], n_landmarks]
        for j in range(n_landmarks):
            x, y = pred_coords_np[i, j]
            row.extend([int(x),int(y)])
        pred_data.append(row)

    pred_columns = ["image_name", "image_width", "image_height", "n_landmarks"]
    for j in range(n_landmarks):
        pred_columns += [f"landmark_{j+1}_x", f"landmark_{j+1}_y"]

    pred_df = pd.DataFrame(pred_data, columns=pred_columns)

    pred_csv_path = os.path.join("test_results", f"{args.experiment_name}_predicted_coordinates.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"✅ Saved predicted coordinates to {pred_csv_path}")


def test_model_wo_label(args, model, device, test_loader):
    """
    Run model inference on test set without ground truth labels.

    Args:
        args (Namespace): Configuration arguments.
        model (nn.Module): Trained model.
        device (str): Device ('cuda' or 'cpu').
        test_loader (DataLoader): Test set dataloader.
    """
    model.eval()
    all_pred_coords = []

    with torch.no_grad():
        for idx, (images, image_name) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            outputs = model(images)

            # Predict coordinates
            probs = torch.sigmoid(outputs)
            B, C, H, W = probs.shape
            probs_flat = probs.view(B, C, -1)
            max_indices = probs_flat.argmax(dim=2)

            pred_coords = torch.zeros((B, C, 2), device=device)
            for b in range(B):
                for c in range(C):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, W)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            all_pred_coords.append(pred_coords)
            overlay_pred_masks_wo_label(image_name, images, pred_coords)


def test(args, model, device):
    """
    Main test function to route based on label availability.

    Args:
        args (Namespace): Parsed command-line arguments.
        model (nn.Module): Trained U-Net model.
        device (str): Device for computation.
    """
    if args.labels == 'y':
        print("Testing with ground truth labels...")
        os.makedirs('visualization_w_label', exist_ok=True)
        test_loader = dataloader(args)
        test_model_w_label(args, model, device, test_loader)
    else:
        print("Testing without ground truth labels...")
        os.makedirs('visualization_wo_label', exist_ok=True)
        test_loader = dataloader(args)
        test_model_wo_label(args, model, device, test_loader)