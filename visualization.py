"""
visualization.py

Visualization utilities for overlaying predicted and ground truth landmark coordinates
onto input images for both labeled and unlabeled testing scenarios.

Author: Yehyun Suh  
Date: 2025-04-15
"""

import os
import cv2
import torch
import numpy as np


def overlay_pred_masks_w_label(image_name, images, pred_coords, gt_coords):
    """
    Overlay predicted and ground truth landmarks on test images.

    Args:
        image_name (str): Name(s) of the image(s) in the batch.
        images (Tensor): Input images of shape [B, 3, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
    """
    os.makedirs("visualization_w_label", exist_ok=True)

    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        # Denormalize the image (ImageNet stats)
        img = images[b].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        # Draw landmark circles
        for c in range(pred.shape[0]):
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)     # Red = predicted
            cv2.circle(img, (gx, gy), 4, (255, 0, 0), -1)     # Blue = ground truth

        # Save overlayed image
        cv2.imwrite(f"visualization_w_label/{image_name[b]}", img)


def overlay_pred_masks_wo_label(image_name, images, pred_coords):
    """
    Overlay predicted landmarks on test images (no ground truth available).

    Args:
        image_name (str): Name(s) of the image(s) in the batch.
        images (Tensor): Input images of shape [B, 3, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
    """
    os.makedirs("visualization_wo_label", exist_ok=True)

    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()

        # Denormalize the image
        img = images[b].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        # Draw predicted landmark circles
        for c in range(pred.shape[0]):
            px, py = int(pred[c, 0]), int(pred[c, 1])
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)  # Red = predicted

        # Save overlayed image
        cv2.imwrite(f"visualization_wo_label/{image_name[b]}", img)