# TODO: Implement 2 classes one for CT Modality and one for the WSI modality

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader


from random import randint


class VolumeStandardizer:
    def standardize(self, volume, target_depth:int, method:str):
        standardizer = self._get_standardizer(method)
        return standardizer(volume,target_depth)
    
    def _get_standardizer(self,method):
        if method == "padding":
            return self._standardize_padding
        else:
            raise ValueError(method)
        
    def _standardize_padding(self,volume,target_depth):
        """
        Pad the 3D volume to a target depth.
        Args:
        - volume: The input 3D volume (D, H, W).
        - target_depth: The target depth to pad to.
        
        Returns:
        - Padded volume.
        """
        depth, _, _ = volume.shape
        
        if depth < target_depth:
            # Compute the padding required
            pad_before = (target_depth - depth) // 2
            pad_after = target_depth - depth - pad_before
            
            # Pad along the depth dimension
            padded_volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_volume = volume
        
        return padded_volume

class UnimodalCTDataset3D(torch.utils.data.Dataset):
    """Class for loading Unimodal CT dicom scans
    """
    num_classes = 3
    dataset_path = "data/processed/"
    map_classes = {"G1":0,"G2":1,"G3":2}
    
    def __init__(self, split:str,dataset_path:str = None, vol_std_method:str = "padding"):
        """
        Args:
            split (str): Choose between 'train', 'val', 'overfit' split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit', 'all']
        self.items = []
        self.classfreq = {"G1":0, "G2":0, "G3":0}
        self.vol_std_method = vol_std_method
            
        if dataset_path:
            self.dataset_path = dataset_path
        #per ogni riga apro la cartella del paziente e faccio "il loading" dei volumi del paziente (?)
        self.labels = {k.strip(): v.strip() for k, v in (line.split(',') for line in Path(f'{os.path.join(self.dataset_path,"labels.txt")}').read_text().splitlines())}
        self.max_depth = 0 
        with open(f"{os.path.join(self.dataset_path,split)}.txt", "r") as split:
            for row in split:
                row = row.strip()        
                for file in os.listdir(os.path.join(self.dataset_path,'CT/'+row)):
                    #print(file)
                    depth = len(np.load(os.path.join(self.dataset_path,'CT/'+row+"/"+file)))
                    
                    if depth > self.max_depth:
                        self.max_depth = depth
                    self.items.extend([row+"/"+file])
                    self.classfreq[self.labels[row]] += 1     
        self.max_depth        
        
    def __getitem__(self, index):
        """
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "patient", a string of the patient's name 
                 "frame", a numpy float32 array representing the CT scan's frame
                 "label", a number in [0, 2] representing the tumor grade
        """
        
        item = self.items[index] 
        patient_id = item.split("/")[0]
        #print(f"item: {item}")
        item_class = self.map_classes[self.labels[patient_id]]
        
        vol = np.load(os.path.join(self.dataset_path,"CT/"+item))
        vol = VolumeStandardizer().standardize(volume=vol,target_depth=self.max_depth, method = self.vol_std_method) 
        return {
            "patient_id": patient_id,
            "volume": vol,
            "label": item_class
        }
    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.items)
    def stats(self):
        return {
            "length": len(self.items),
            "class_frequency": self.classfreq
        }
    
    @staticmethod
    def move_batch_to_device(batch, device):
        """Utility methof for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['volume'] = batch['volume'].to(device)
        batch['label'] = batch['label'].to(device)


def sanity_check_dataset():
    # Instantiate the dataset
    dataset = UnimodalCTDataset3D(split='val', dataset_path =  "data/processed/")
    print(f"MAX DEPTH: {dataset.max_depth}")
    # Check stats of dataset
    print(f"Dataset stats: {dataset.stats()}")

    # Check the first few items in the dataset
    for i in range(3):
        min = 0
        max = len(dataset)
        item = dataset[(randint(min,max))]
        print(f"Item {i}:")
        print(f"  Patient ID: {item['patient_id']}")
        print(f"  Volume shape: {item['volume'].shape}")
        print(f"  Label: {item['label']}")
    
    # Check if DataLoader works with the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a batch of data
    batch = next(iter(dataloader))
    
    # Check the batch
    print(f"Batch patient IDs: {batch['patient_id']}")
    print(f"Batch Volume shape: {batch['volume'].shape}")
    print(f"Batch labels: {batch['label']}")

    # Move batch to device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.move_batch_to_device(batch, device)
    print(f"Batch moved to device: {device}")

if __name__ == "__main__":
    test_dataset()