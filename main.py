"""
main.py

Testing pipeline for anatomical landmark detection using U-Net.

This script parses command-line arguments, prepares the testing environment,
loads the model, and initiates the testing loop.

Author: Yehyun Suh  
Date: 2025-04-15  
Copyright: (c) 2025 Yehyun Suh

Example:
    python main.py \
        --test_image_dir ./data/test_images \
        --label_dir ./data/labels \
        --test_csv_file test_annotation.csv \
        --image_resize 512 \
        --n_landmarks 2 \
        --labels y \
        --weight_dir ./weight \
        --weight_name best_model.pth
"""

import os
import torch
import argparse

from utils import customize_seed
from model import UNet
from test import test


def main(args):
    """
    Main function that initializes and tests the model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UNet(args.n_landmarks, device)

    # Load model weights
    weight_path = os.path.join(args.weight_dir, args.weight_name)
    if os.path.exists(weight_path):
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
        print(f"✅ Model weights loaded from {weight_path}")
    else:
        raise FileNotFoundError(f"❌ Weight file not found: {weight_path}")

    model.to(device)
    print("✅ Model loaded successfully.")

    # Run the testing procedure
    test(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the anatomical landmark detection model on test images."
    )

    # Reproducibility
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    # Data settings
    parser.add_argument(
        '--test_image_dir', type=str, default='./data/test_images',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--label_dir', type=str, default='./data/labels',
        help='Directory containing label annotations'
    )
    parser.add_argument(
        '--test_csv_file', type=str, default='test_annotation.csv',
        help='CSV file listing test images and landmark coordinates'
    )
    parser.add_argument(
        '--image_resize', type=int, default=512,
        help='Resize image size (must be divisible by 32)'
    )
    parser.add_argument(
        '--n_landmarks', type=int, required=True,
        help='Number of landmarks per image'
    )

    # Testing settings
    parser.add_argument(
        '--labels', type=str, default=None, required=True,
        help='Set to "y" if labels exist, "n" otherwise'
    )
    parser.add_argument(
        '--weight_dir', type=str, default='./weight',
        help='Directory containing saved model weights'
    )
    parser.add_argument(
        '--weight_name', type=str, default='best_model.pth',
        help='Filename of the weight to load'
    )

    args = parser.parse_args()

    # Fix seed for reproducibility
    customize_seed(args.seed)

    # Start main
    main(args)