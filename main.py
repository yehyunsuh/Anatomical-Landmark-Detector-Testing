"""
main.py

Testing pipeline for anatomical landmark detection using U-Net.

This script parses command-line arguments, prepares the testing environment, loads the model,
and initiates the testing loop.

Author: Yehyun Suh  
Date: 2025-04-15
Copyright: (c) 2025 Yehyun Suh

Example:
    python main.py
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

    # Load the model weights
    weight_path = os.path.join(args.weight_dir, args.weight_name)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print(f"Model weights loaded from {weight_path}")
    else:
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    model.to(device)
    print("Model loaded successfully.")

    # Start testing
    test(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for the project.")

    # Arguments for reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    # Arguments for data
    parser.add_argument('--test_image_dir', type=str, default='./data/test_images', help='Directory for images')
    parser.add_argument('--label_dir', type=str, default='./data/labels', help='Directory for labels (annotations)')
    parser.add_argument('--test_csv_file', type=str, default='test_annotation.csv', help='Name of the CSV file with image and label names')
    parser.add_argument('--image_resize', type=int, default=512, help='Size of the images after resizing, it should be divisible by 32')
    parser.add_argument('--n_landmarks', type=int, default=2, required=True, help='Number of landmarks in the image')
    
    # Arguments for testing
    parser.add_argument('--labels', type=str, default=None, required=True, help='Whether you have labels for the test set or not, answer with "y" or "n"')
    parser.add_argument('--weight_dir', type=str, default='./weight', help='Directory for model weights')
    parser.add_argument('--weight_name', type=str, default='best_model.pth', help='Name of the weight file')
    
    args = parser.parse_args()
    
    # Fix seed for reproducibility
    customize_seed(args.seed)
    
    main(args)