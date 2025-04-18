"""
model.py

Defines the U-Net segmentation model for anatomical landmark detection.

Author: Yehyun Suh  
Date: 2025-04-15  
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


def UNet(n_landmarks, device):
    """
    Constructs and returns a U-Net model using a ResNet-101 encoder,
    configured for anatomical landmark detection.

    Args:
        n_landmarks (int): Number of landmarks (output channels).
        device (str): Device to move the model to ('cuda' or 'cpu').

    Returns:
        nn.Module: Configured U-Net model.
    """
    print("---------- Loading Model ----------")

    model = smp.Unet(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        classes=n_landmarks,
        activation='sigmoid',  # Removed below for logits-based loss
    )

    print("---------- Model Loaded ----------")

    # Remove final sigmoid to use BCEWithLogitsLoss instead
    model.segmentation_head = nn.Sequential(
        *list(model.segmentation_head.children())[:-1]
    )

    return model.to(device)