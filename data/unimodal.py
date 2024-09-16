# TODO: Implement 2 classes one for CT Modality and one for the WSI modality

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader


from random import randint
class UnimodalCTDataset(torch.utils.data.Dataset):
    """Class for loading Unimodal CT dicom scans
    """
    num_classes = 3
    dataset_path = "data/processed/"
    map_classes = {"G1":0,"G2":1,"G3":2}
    classfreq = {"G1":0, "G2":0, "G3":0}
    def __init__(self, split:str,dataset_path:str = None):
        """
        Args:
            split (str): Choose between 'train', 'val', 'overfit' split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.items = []
        if dataset_path:
            self.dataset_path = dataset_path
        #per ogni riga apro la cartella del paziente e faccio "il loading" dei volumi del paziente (?)
        self.labels = {k.strip(): v.strip() for k, v in (line.split(',') for line in Path(f'{self.dataset_path}labels.txt').read_text().splitlines())}
         
        with open(f"{self.dataset_path}{split}.txt", "r") as split:
            for row in split:
                row = row.strip()
                for file in os.listdir(self.dataset_path+'CT/'+row):
                    #print(file)
                    npy_ct = np.load(self.dataset_path+"CT/"+row+"/"+file)
                    self.classfreq[self.labels[row]] += len(npy_ct)
                    self.items.extend([row+"/"+file+"_"+str(i) for i in range(len(npy_ct))])     
        #self.items = Path(f"data/processed/{split}.txt").read_text().splitlines()
        
        
    def __getitem__(self, index):
        """
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "patient", a string of the patient's name 
                 "frame", a numpy float32 array representing the CT scan's frame
                 "label", a number in [0, 2] representing the tumor grade
        """
        
        item = self.items[index] 
        patient_id = item.split("_")[0].split("/")[0]
        item_class = self.map_classes[self.labels[patient_id]]
        
        scan_frame = np.load(self.dataset_path+"CT/"+item.split("_")[0])[int(item.split("_")[1])]
        scan_frame = np.stack([scan_frame] * 3, axis=0)
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
        batch['frame'] = batch['frame'].to(device)
        batch['label'] = batch['label'].to(device)




def test_dataset():
    # Instantiate the dataset
    dataset = UnimodalCTDataset(split='train', dataset_path =  "data/processed/")

    # Check stats of dataset
    print(f"Dataset stats: {dataset.stats()}")

    # Check the first few items in the dataset
    for i in range(3):
        min = 0
        max = len(dataset)
        item = dataset[(randint(min,max))]
        print(f"Item {i}:")
        print(f"  Patient ID: {item['patient_id']}")
        print(f"  Frame shape: {item['frame'].shape}")
        print(f"  Label: {item['label']}")
    
    # Check if DataLoader works with the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a batch of data
    batch = next(iter(dataloader))
    
    # Check the batch
    print(f"Batch patient IDs: {batch['patient_id']}")
    print(f"Batch frame shape: {batch['frame'].shape}")
    print(f"Batch labels: {batch['label']}")

    # Move batch to device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.move_batch_to_device(batch, device)
    print(f"Batch moved to device: {device}")

if __name__ == "__main__":
    test_dataset()