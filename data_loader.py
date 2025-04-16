"""
data_loader.py

Dataset and dataloader utilities for anatomical landmark segmentation.

Author: Yehyun Suh
"""

import os
import csv
import cv2
import torch
import numpy as np
import albumentations as A

from scipy.ndimage import binary_dilation
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split


class SegmentationDataset(Dataset):
    """
    Custom dataset for anatomical landmark segmentation.
    Each sample includes an RGB image and a multi-channel binary mask,
    where each channel corresponds to a dilated landmark point.
    """

    def __init__(self, csv_path, image_dir, n_landmarks=None):
        """
        Initializes the dataset by parsing CSV annotations and storing image/landmark paths.

        Args:
            csv_path (str): Path to the annotation CSV file.
            image_dir (str): Directory containing input images.
            n_landmarks (int): Number of landmarks per image.
        """
        self.image_dir = image_dir
        self.samples = []
        self.n_landmarks = n_landmarks

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                image_name = row[0]
                coords = list(map(int, row[4:]))
                assert len(coords) == 2 * n_landmarks, "Mismatch in number of landmark coordinates"
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                self.samples.append((image_name, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, landmarks = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        max_side = max(h, w)

        # Apply resizing and normalization
        transform = A.Compose([
            A.PadIfNeeded(min_height=max_side, min_width=max_side),
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        # transform = A.Compose([
        #     A.PadIfNeeded(min_height=max_side, min_width=max_side),
        #     A.Resize(512, 512),
        #     A.Normalize(
        #         mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225),
        #     ),
        #     ToTensorV2()
        # ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        transformed = transform(image=image, keypoints=landmarks)
        image = transformed['image']  # Tensor: [3, H, W]
        new_landmarks = transformed['keypoints']

        return image, image_name, new_landmarks
    

class SegmentationDataset_wo_Label(Dataset):
    """
    Custom dataset for anatomical landmark segmentation.
    Each sample includes an RGB image and a multi-channel binary mask,
    where each channel corresponds to a dilated landmark point.
    """

    def __init__(self, csv_path, image_dir, n_landmarks=None):
        """
        Initializes the dataset by parsing CSV annotations and storing image paths.

        Args:
            csv_path (str): Path to the annotation CSV file.
            image_dir (str): Directory containing input images.
        """
        self.image_dir = image_dir
        self.samples = []
        self.n_landmarks = n_landmarks

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                image_name = row[0]
                self.samples.append((image_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        max_side = max(h, w)

        # Apply resizing and normalization
        transform = A.Compose([
            A.PadIfNeeded(min_height=max_side, min_width=max_side),
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        # transform = A.Compose([
        #     A.PadIfNeeded(min_height=max_side, min_width=max_side),
        #     A.Resize(512, 512),
        #     A.Normalize(
        #         mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225),
        #     ),
        #     ToTensorV2()
        # ])

        transformed = transform(image=image)
        image = transformed['image']  # Tensor: [3, H, W]

        return image, image_name


def dataloader(args):
    """
    Constructs and returns PyTorch dataloaders for testing.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing all config.

    Returns:
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    if args.labels == 'y':
        dataset = SegmentationDataset(
            csv_path=os.path.join(args.label_dir, args.test_csv_file),
            image_dir=args.test_image_dir,
            n_landmarks=args.n_landmarks,
        )

        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        print(f"Test size: {len(test_loader.dataset)}")

        return test_loader
    else:
        # TODO: Create .csv file for test set and save it in the label directory
        image_names = os.listdir(args.test_image_dir)
        os.makedirs(args.label_dir, exist_ok=True)
        with open(os.path.join(args.label_dir, args.test_csv_file), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'n_landmarks'])
            for image_name in image_names:
                writer.writerow([image_name, args.n_landmarks])

        dataset = SegmentationDataset_wo_Label(
            csv_path=os.path.join(args.label_dir, args.test_csv_file),
            image_dir=args.test_image_dir,
            n_landmarks=args.n_landmarks,
        )

        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        print(f"Test size: {len(test_loader.dataset)}")

        return test_loader