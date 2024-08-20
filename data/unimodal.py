# TODO: Implement 2 classes one for CT Modality and one for the WSI modality

from pathlib import Path

import numpy as np
import torch

class UnimodalCTDataset(torch.utils.data.Dataset):
    """Class for loading Unimodal CT dicom scans
    """
    num_classes = 3
    dataset_path = Path("data/processed/CT")
    map_classes = {"G1":0,"G2":1,"G3":2}
    
    def __init__(self, split:str):
        """
        Args:
            split (str): Choose between 'train', 'val', 'overfit' split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        with open(f"data/processed/{split}.txt", "r") as split:
            for row in split:
                self.items.append([row+"_"+i for i in range(len(np.load(self.dataset_path+"/"+row+".npy")))])     
        #self.items = Path(f"data/processed/{split}.txt").read_text().splitlines()
        self.labels = {k.strip(): v.strip() for k, v in (line.split(',') for line in Path('data/processed/labels.txt').read_text().splitlines())}
        
        
    def __getitem__(self, index):
        """
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "patient", a string of the patient's name 
                 "frame", a numpy float32 array representing the CT scan's frame
                 "label", a number in [0, 2] representing the tumor grade
        """
        # TODO Get item associated with index, get class, load voxels with ShapeNetVox.get_shape_voxels
        
        item = self.items[index] 
        # Hint: since shape names are in the format "<shape_class>/<shape_identifier>", the first part gives the class
        patient_id = item.split("_")[0]
        item_class = self.map_classes[self.labels[patient_id]]
        
        # read voxels from binvox format on disk as 3d numpy arrays
        scan_frame = np.load(self.dataset_path+"/"+patient_id+".npy")[item.split("_")[1]]
        return {
            "patient_id": patient_id,
            "frame": scan_frame,
            "label": item_class
        }
    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.items)
    @staticmethod
    def move_batch_to_device(batch, device):
        """Utility methof for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['frame'] = batch['frame'].to(device)
        batch['label'] = batch['label'].to(device)
