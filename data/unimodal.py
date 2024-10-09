# TODO: Implement 2 classes one for CT Modality and one for the WSI modality

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from random import randint
class UnimodalCTDataset(torch.utils.data.Dataset):
    """Class for loading Unimodal CT dicom scans
    """
    num_classes = 3
    dataset_path = "data/processed/"
    map_classes = {"G1":0,"G2":1,"G3":2}
    
    def __init__(self, split:str,dataset_path:str = None, transform = None):
        """
        Args:
            split (str): Choose between 'train', 'val', 'overfit' split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit', 'all']
        self.items = []
        self.classfreq = {"G1":0, "G2":0, "G3":0}
        if dataset_path:
            self.dataset_path = dataset_path
        #per ogni riga apro la cartella del paziente e faccio "il loading" dei volumi del paziente (?)
        self.labels = {k.strip(): v.strip() for k, v in (line.split(',') for line in Path(f'{os.path.join(self.dataset_path,"labels.txt")}').read_text().splitlines())}
        self.transform = transform
        with open(f"{os.path.join(self.dataset_path,split)}.txt", "r") as split:
            for row in split:
                row = row.strip()
                for file in os.listdir(os.path.join(self.dataset_path,'CT/'+row)):
                    #print(file)
                    npy_ct = np.load(os.path.join(self.dataset_path,'CT/'+row+"/"+file))
                    self.classfreq[self.labels[row]] += len(npy_ct)
                    self.items.extend([row+"/"+file+"_"+str(i) for i in range(len(npy_ct))])     
        #self.items = Path(f"data/processed/{split}.txt").read_text().splitlines()
        self.weights = self.calculate_weights()
        
    def calculate_weights(self):
        """Calculates weights for each sample in the dataset based on class frequencies."""
        weights = []
        total_samples = sum(self.classfreq.values())
        for item in self.items:
            patient_id = item.split("_")[0].split("/")[0]
            label = self.map_classes[self.labels[patient_id]]
            # Calculate weight for the sample
            weight = total_samples / (self.classfreq[self.labels[patient_id]] * self.num_classes)
            weights.append(weight)
        return weights
        
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
        
        scan_frame = np.load(os.path.join(self.dataset_path,"CT/"+item.split("_")[0]))[int(item.split("_")[1])]
        scan_frame = np.stack([scan_frame] * 3, axis=0)
        
        # Apply transformations if provided
        if self.transform:
            scan_frame = np.transpose(scan_frame, (1,2,0))
            scan_frame = self.transform(scan_frame)
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
    
    
    def compute_subset_weights(self, subset_indices):
        """Static method to create a weighted sampler for a given subset of the dataset."""
        # Get the labels of the subset items using the original dataset
        subset_labels = [self.items[i] for i in subset_indices]

        # Calculate the class frequency for the subset
        class_counts = {k: 0 for k in self.map_classes.values()}
        for item in subset_labels:
            patient_id = item.split("_")[0].split("/")[0]
            label = self.labels[patient_id]
            class_counts[self.map_classes[label]] += 1

        # Calculate class weights for the subset
        total_samples = len(subset_indices)
        class_weights = {k: total_samples / (v * len(class_counts)) if v > 0 else 0 for k, v in class_counts.items()}

        # Assign a weight to each sample in the subset based on its class
        subset_weights = []
        for item in subset_labels:
            patient_id = item.split("_")[0].split("/")[0]
            label = self.labels[patient_id]
            subset_weights.append(class_weights[self.map_classes[label]])
        return subset_weights




def sanity_check_dataset():

    transform = transforms.Compose([
        transforms.ToTensor(),    
        # Apply a random subset of the following transformations
        transforms.RandomApply([
            transforms.RandomRotation(degrees=10),       # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
            transforms.GaussianBlur(kernel_size=(5,5),sigma=(0.1,0.5))
        ], p=0.5),  # Only apply the above transformations with a probability of 0.5
    ])
    # Instantiate the dataset
    dataset = UnimodalCTDataset(split='all', dataset_path =  "data/processed", transform = transform)

    # Create a WeightedRandomSampler using the calculated weights
    sampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset.weights), replacement=True)

    
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
    sanity_check_dataset()