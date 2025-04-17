"""
data_loader.py

Dataset and dataloader utilities for anatomical landmark detection.

Author: Yehyun Suh  
Date: 2025-04-15
"""

import os
import csv
import cv2
import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):
    """
    Dataset class for test images with ground truth landmark annotations.
    Each sample includes an image and a list of keypoint coordinates.
    """

    def __init__(self, csv_path, image_dir, n_landmarks=None):
        """
        Initialize the dataset by reading image names and landmark coords.

        Args:
            csv_path (str): Path to CSV containing image and landmark data.
            image_dir (str): Path to directory containing test images.
            n_landmarks (int): Number of landmarks per image.
        """
        self.image_dir = image_dir
        self.samples = []
        self.n_landmarks = n_landmarks

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                image_name = row[0]
                coords = list(map(int, row[4:]))
                assert len(coords) == 2 * n_landmarks, \
                    f"Mismatch in number of landmarks: expected {2*n_landmarks}, got {len(coords)}"
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                self.samples.append((image_name, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, landmarks = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        max_side = max(h, w)

        transform = A.Compose([
            A.PadIfNeeded(min_height=max_side, min_width=max_side),
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        transformed = transform(image=image, keypoints=landmarks)
        image = transformed['image']
        new_landmarks = transformed['keypoints']

        return image, image_name, new_landmarks


class SegmentationDataset_wo_Label(Dataset):
    """
    Dataset class for test images without ground truth landmarks.
    Returns image tensor and filename only.
    """

    def __init__(self, csv_path, image_dir, n_landmarks=None):
        """
        Initialize dataset with image file names only.

        Args:
            csv_path (str): Path to CSV listing test image names.
            image_dir (str): Directory containing test images.
            n_landmarks (int): Number of landmarks (not used, but kept for symmetry).
        """
        self.image_dir = image_dir
        self.samples = []
        self.n_landmarks = n_landmarks

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                image_name = row[0]
                self.samples.append(image_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        max_side = max(h, w)

        transform = A.Compose([
            A.PadIfNeeded(min_height=max_side, min_width=max_side),
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        transformed = transform(image=image)
        image = transformed['image']

        return image, image_name


def dataloader(args):
    """
    Constructs the PyTorch dataloader for labeled or unlabeled test data.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        DataLoader: The test dataloader.
    """
    csv_path = os.path.join(args.label_dir, args.test_csv_file)
    image_dir = args.test_image_dir

    if args.labels == 'y':
        dataset = SegmentationDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            n_landmarks=args.n_landmarks,
        )
    else:
        # If no label CSV exists, auto-generate from image filenames
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            print("Generating CSV file from image directory...")
            os.makedirs(args.label_dir, exist_ok=True)
            image_names = os.listdir(image_dir)

            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['image_name', 'n_landmarks'])
                for name in image_names:
                    writer.writerow([name, args.n_landmarks])
        else:
            print(f"CSV file already exists: {csv_path}")

        dataset = SegmentationDataset_wo_Label(
            csv_path=csv_path,
            image_dir=image_dir,
            n_landmarks=args.n_landmarks,
        )

    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"🧪 Test size: {len(test_loader.dataset)}")

    return test_loader