"""
utils.py

Utility functions for model reproducibility.

Author: Yehyun Suh  
Date: 2025-04-15  
"""

import torch
import random
import numpy as np


def customize_seed(seed):
    """
    Set seeds across libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to use for torch, numpy, and random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Uncomment below if using multi-GPU setup
    # torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)