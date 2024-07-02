# TODO: Implement 2 classes one for CT Modality and one for the WSI modality

from pathlib import Path

import numpy as np
import torch

class UnimodalCTDataset(torch.utils.data.Dataset):
    """Class for loading Unimodal CT dicom scans
    """
    def __init__(self, split:str):
        """
        Args:
            split (str): Choose between 'train', 'val', 'overfit' split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        
        