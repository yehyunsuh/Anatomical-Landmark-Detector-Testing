"""
visualization.py

Visualization utilities for overlaying predicted/ground truth landmark masks.

Author: Yehyun Suh
"""

import os
import cv2
import torch
import numpy as np


def overlay_pred_masks_w_label(image_name, images, pred_coords, gt_coords):
    """
    Overlay predicted masks and landmarks per landmark channel on the image.

    Args:
        image_name (str): Name of the image.
        images (Tensor): Batch of images [B, 3, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
    """
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        img = (img * imagenet_std + imagenet_mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        for c in range(pred.shape[0]):
            # Draw landmark circles
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)  # Red = predicted
            cv2.circle(img, (gx, gy), 4, (255, 0, 0), -1)  # Green = GT

        cv2.imwrite(f'visualization_w_label/{image_name[b]}', img)


def overlay_pred_masks_wo_label(image_name, images, pred_coords):
    """
    Overlay predicted masks and landmarks per landmark channel on the image.

    Args:
        image_name (str): Name of the image.
        images (Tensor): Batch of images [B, 3, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
    """
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        img = (img * imagenet_std + imagenet_mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        # Draw landmark circles
        for c in range(pred.shape[0]):
            px, py = int(pred[c, 0]), int(pred[c, 1])
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)  # Red = predicted

        cv2.imwrite(f'visualization_wo_label/{image_name[b]}', img)