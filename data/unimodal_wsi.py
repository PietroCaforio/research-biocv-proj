import glob
import os
from pathlib import Path
from random import randint

import torch
from PIL import PngImagePlugin
from torch.utils.data import DataLoader
from torchvision.io import read_image

# import numpy as np
# from PIL import Image

PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 100
# from torch.utils.data import WeightedRandomSampler
# from torchvision import transforms


class UnimodalWSIDataset(torch.utils.data.Dataset):
    """
    Class for loading Unimodal WSI scans
    """

    num_classes = 3
    dataset_path = "data/processed/"
    map_classes = {"G1": 0, "G2": 1, "G3": 2}

    def __init__(self, split: str, dataset_path: str = None, transform=None):
        """
        Args:
            split (str): Choose between 'train', 'val', 'overfit' split
        """
        super().__init__()
        assert split in ["train", "val", "overfit", "all"]
        self.items = []
        self.classfreq = {"G1": 0, "G2": 0, "G3": 0}
        if dataset_path:
            self.dataset_path = dataset_path
        # Generate a dictionary from the labels file where key
        # is the patient_id and value is the actual label (G1, G2, G3)
        self.labels = {
            k.strip(): v.strip()
            for k, v in (
                line.split(",")
                for line in Path(f'{os.path.join(self.dataset_path,"labels.txt")}')
                .read_text()
                .splitlines()
            )
        }
        self.transform = transform
        with open(f"{os.path.join(self.dataset_path,split)}.txt") as split:  # type: ignore
            for row in split:
                row = row.strip()  # patient_id
                wsi_path = os.path.join(self.dataset_path, "WSI/")
                for folder in glob.glob(f"{wsi_path}/*{row}*"):
                    # Iterate through the patch files in the slide folder
                    for file in os.listdir(folder):
                        # Add them to the items list as a dictionary with
                        # "slide_folder", "patient_id", "np.array as image"
                        self.items.append(
                            {"patient_id": row, "slide_folder": folder, "patch": file}
                        )
                        # Update the classfrequency
                        self.classfreq[
                            self.labels[row]
                        ] += 1  # the class frequency is given by the #patches per class
        self.weights = self.calculate_weights()

    def calculate_weights(self):
        """Calculates weights for each sample in the dataset based on class frequencies."""
        weights = []
        total_samples = sum(self.classfreq.values())
        for item in self.items:
            patient_id = item["patient_id"]
            # Calculate weight for the sample
            weight = total_samples / (
                self.classfreq[self.labels[patient_id]] * self.num_classes
            )
            weights.append(weight)
        return weights

    def __getitem__(self, index):
        """
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape with keys:
                 "patient", a string of the patient's name
                 "frame", a torch tensor of the WSI patch
                 "label", a number in [0, 2] representing the tumor grade
        """
        item = self.items[index]
        patient_id = item["patient_id"]
        item_class = self.map_classes[self.labels[patient_id]]

        # patch = np.array(
        #    Image.open(os.path.join(item["slide_folder"], item["patch"]))
        # ).transpose(2, 0, 1)
        patch = read_image(os.path.join(item["slide_folder"], item["patch"]))

        return {
            "patient_id": patient_id,
            "patch": patch,
            "label": item_class,
            "slide": item["slide_folder"],
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.items)

    def stats(self):
        wsi_per_patient = {}
        # patch_counts = {}

        for item in self.items:

            if item["patient_id"] not in wsi_per_patient:
                wsi_per_patient[item["patient_id"]] = set()
            wsi_per_patient[item["patient_id"]].add(item["slide_folder"])
        return {
            "total_patients": len(wsi_per_patient),
            "length": len(self.items),
            "class_frequency": self.classfreq,
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        """Utility methof for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch["patch"] = batch["patch"].to(device)
        batch["label"] = batch["label"].to(device)

    def compute_subset_weights(self, subset_indices):
        """Static method to create a weighted sampler for a given subset of the dataset."""
        # Get the labels of the subset items using the original dataset
        subset_labels = [self.items[i] for i in subset_indices]

        # Calculate the class frequency for the subset
        class_counts = {k: 0 for k in self.map_classes.values()}
        for item in subset_labels:
            patient_id = item["patient_id"]
            label = self.labels[patient_id]
            class_counts[self.map_classes[label]] += 1

        # Calculate class weights for the subset
        total_samples = len(subset_indices)
        class_weights = {
            k: total_samples / (v * len(class_counts)) if v > 0 else 0
            for k, v in class_counts.items()
        }

        # Assign a weight to each sample in the subset based on its class
        subset_weights = []
        for item in subset_labels:
            patient_id = item[patient_id]
            label = self.labels[patient_id]
            subset_weights.append(class_weights[self.map_classes[label]])
        return subset_weights


def sanity_check_dataset():

    # transform = transforms.Compose(
    #    [
    #        transforms.ToTensor(),
    #        # Apply a random subset of the following transformations
    #        transforms.RandomApply(
    #            [
    #                transforms.RandomRotation(degrees=10),  # Random rotation
    #                transforms.ColorJitter(
    #                    brightness=0.2, contrast=0.2
    #                ),  # Adjust brightness/contrast
    #                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)),
    #            ],
    #            p=0.5,
    #        ),  # Only apply the above transformations with a probability of 0.5
    #    ]
    # )
    # Instantiate the dataset
    dataset = UnimodalWSIDataset(
        split="all",
        dataset_path="data/processed/processed_CPTAC_PDA_71_3D",
        transform=None,
    )

    # Create a WeightedRandomSampler using the calculated weights
    # sampler = WeightedRandomSampler(
    #    weights=dataset.weights, num_samples=len(dataset.weights), replacement=True
    # )

    # Check stats of dataset
    print(f"Dataset stats: {dataset.stats()}")

    # for i in range(len(dataset)):
    #    item = dataset[i]
    # Check the first few items in the dataset
    for i in range(3):
        min = 0
        max = len(dataset)
        item = dataset[(randint(min, max))]
        print(f"Item {i}:")
        print(f"  Patient ID: {item['patient_id']}")
        print(f"  Frame shape: {item['patch'].shape}")
        print(f"  Label: {item['label']}")

    # Check if DataLoader works with the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get a batch of data
    batch = next(iter(dataloader))

    # Check the batch
    print(f"Batch patient IDs: {batch['patient_id']}")
    print(f"Batch frame shape: {batch['patch'].shape}")
    print(f"Batch labels: {batch['label']}")

    # Move batch to device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.move_batch_to_device(batch, device)
    print(f"Batch moved to device: {device}")


if __name__ == "__main__":
    sanity_check_dataset()
# C3L-02888
# C3L-03348
# C3L-03622
# C3L-03629
# C3N-01716 -> lui c'é ma è problematico e quindi nel dataset processato
#             non apparirà alla fine
# Mancano dal dataset di WSI di CPTAC-PDA per cui CPTAC-PDA-71 in realtà
# sarà di 66 pazienti nel caso multimodale
