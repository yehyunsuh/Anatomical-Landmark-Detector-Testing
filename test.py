import os
import torch
import pandas as pd

from tqdm import tqdm

from data_loader import dataloader
from visualization import overlay_pred_masks_w_label, overlay_pred_masks_wo_label


def test_model_w_label(args, model, device, test_loader):
    model.eval()
    all_pred_coords = []
    all_gt_coords = []

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

            overlay_pred_masks_w_label(image_name, images, pred_coords, gt_coords)

    all_pred_coords = torch.cat(all_pred_coords, dim=0)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)
    dists = torch.norm(all_pred_coords - all_gt_coords, dim=2)
    mean_dist = dists.mean().item()

    # Calculate mean and std per image
    dist_mean = dists.mean(dim=1).cpu().numpy()
    dist_std = dists.std(dim=1).cpu().numpy()
    dists_np = dists.cpu().numpy()
    n_landmarks = dists.shape[1]

    # Generate image names (you can customize if you have actual names)
    image_names = [f"img_{i:03d}.png" for i in range(len(dists_np))]

    # Build rows
    data = []
    for i in range(len(image_names)):
        row = [image_names[i], n_landmarks, dist_mean[i], dist_std[i]] + dists_np[i].tolist()
        data.append(row)

    # Create column names
    columns = ["image_name", "n_landmarks", "dist_mean", "dist_std"]
    columns += [f"dist_landmark{j+1}" for j in range(n_landmarks)]

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Compute overall averages
    overall_mean = dist_mean.mean()
    overall_std = dist_std.mean()
    overall_per_landmark = dists_np.mean(axis=0)  # shape: (n_landmarks,)

    # Create final summary row
    summary_row = ["overall average", n_landmarks, overall_mean, overall_std] + overall_per_landmark.tolist()

    # Append to DataFrame
    df.loc[len(df.index)] = summary_row

    # Save CSV
    os.makedirs("test_results", exist_ok=True)
    csv_path = os.path.join("test_results", "landmark_distances.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved landmark distances to {csv_path}")


def test_model_wo_label(args, model, device, test_loader):
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
    if args.labels == 'y':
        print("Testing with labels...")
        os.makedirs('visualization_w_label', exist_ok=True)
        test_loader = dataloader(args)
        test_model_w_label(args, model, device, test_loader)
    else:
        print("Testing without labels...")
        os.makedirs('visualization_wo_label', exist_ok=True)
        test_loader = dataloader(args)
        test_model_wo_label(args, model, device, test_loader)