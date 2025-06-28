import glob
import os
import random
from pathlib import Path
from random import randint

import torch
from PIL import PngImagePlugin
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image

# Increase PNG chunk size limit
PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 100


class UnimodalWSIDataset3D(Dataset):
    """Dataset class for loading multiple WSIs per patient"""

    num_classes = 3
    map_classes = {"G1": 0, "G2": 1, "G3": 2}

    def __init__(
        self,
        split: str,
        dataset_path: str,
        patches_per_wsi: int = 100,
        sampling_strategy: str = "random",  # Options: "random" or "consecutive"
    ):
        """
        Args:
            split (str): Choose between 'train', 'val', 'overfit', 'all'
            dataset_path (str): Path to the dataset root directory
            patches_per_wsi (int): Number of patches to sample per WSI
            sampling_strategy (str): How to sample patches - "random" or "consecutive"
        """
        super().__init__()
        assert split in ["train", "val", "overfit", "all"]
        assert sampling_strategy in [
            "random",
            "consecutive",
        ], "Sampling strategy must be 'random' or 'consecutive'"

        self.dataset_path = dataset_path
        self.patches_per_wsi = patches_per_wsi
        self.sampling_strategy = sampling_strategy

        # Initialize containers
        self.items = []  # Will store (patient_id, wsi_folder, [patch_paths]) tuples
        self.classfreq = {"G1": 0, "G2": 0, "G3": 0}

        # Load labels
        self.labels = {
            k.strip(): v.strip()
            for k, v in (
                line.split(",")
                for line in Path(f'{os.path.join(dataset_path, "labels.txt")}')
                .read_text()
                .splitlines()
            )
        }

        # Load split file and organize patches by patient and WSI
        wsi_path = os.path.join(self.dataset_path, "WSI/")
        with open(f"{os.path.join(dataset_path, split)}.txt") as split_file:
            for row in split_file:
                patient_id = row.strip()

                # Find all WSI folders for this patient
                for wsi_folder in glob.glob(f"{wsi_path}/*{patient_id}*"):
                    # Get all patches for this WSI
                    patches = sorted(
                        # Sort patches to ensure consistent ordering for consecutive sampling
                        f
                        for f in os.listdir(wsi_folder)
                        if f.endswith((".png", ".jpg", ".jpeg"))
                    )

                    if patches:  # Only add if WSI has patches
                        self.items.append((patient_id, wsi_folder, patches))
                        self.classfreq[self.labels[patient_id]] += 1

        self.weights = self.calculate_weights()

    def calculate_weights(self):
        """Calculates weights for each sample in the dataset based on class frequencies."""
        weights = []
        total_samples = sum(self.classfreq.values())

        for patient_id, _, _ in self.items:
            weight = total_samples / (
                self.classfreq[self.labels[patient_id]] * self.num_classes
            )
            weights.append(weight)
        return weights

    def _sample_patches(self, patches):
        """
        Sample patches according to the chosen strategy

        Args:
            patches (list): List of patch filenames

        Returns:
            list: Selected patch filenames
        """
        if len(patches) <= self.patches_per_wsi:
            # If we don't have enough patches, repeat the existing ones
            return patches * (self.patches_per_wsi // len(patches)) + random.sample(
                patches, self.patches_per_wsi % len(patches)
            )

        if self.sampling_strategy == "random":
            # Random sampling
            return random.sample(patches, self.patches_per_wsi)
        else:  # consecutive
            # Choose a random starting point and take consecutive patches
            max_start = len(patches) - self.patches_per_wsi
            start_idx = random.randint(0, max_start)
            return patches[start_idx : start_idx + self.patches_per_wsi]  # noqa E203

    def __getitem__(self, index):
        """
        Returns:
            dict:{
                'patient_id': str,
                'volume': tensor of shape (3, N, H, W) where N is patches_per_wsi,
                'label': int,
                'slide': str
            }
        """
        patient_id, wsi_folder, patch_files = self.items[index]

        # Sample patches according to strategy
        selected_patches = self._sample_patches(patch_files)

        # Load patches
        patches = []
        for patch_file in selected_patches:
            patch_path = os.path.join(wsi_folder, patch_file)
            img_tensor = read_image(patch_path)  # Shape: (3, H, W)
            patches.append(img_tensor)

        # Stack patches into a volume
        volume = torch.stack(patches, dim=1)  # Shape: (3, N, H, W)

        return {
            "patient_id": patient_id,
            "volume": volume,
            "label": self.map_classes[self.labels[patient_id]],
            "slide": wsi_folder,
        }

    def __len__(self):
        return len(self.items)

    def stats(self):
        wsi_per_patient = {}
        patch_counts = {}

        for patient_id, wsi_folder, patches in self.items:
            if patient_id not in wsi_per_patient:
                wsi_per_patient[patient_id] = set()
            wsi_per_patient[patient_id].add(wsi_folder)
            patch_counts[wsi_folder] = len(patches)

        return {
            "total_wsis": len(self.items),
            "total_patients": len(wsi_per_patient),
            "class_frequency": self.classfreq,
            "patches_per_wsi": self.patches_per_wsi,
            "sampling_strategy": self.sampling_strategy,
            "min_patches_in_wsi": min(patch_counts.values()) if patch_counts else 0,
            "max_patches_in_wsi": max(patch_counts.values()) if patch_counts else 0,
            "avg_patches_per_wsi": sum(patch_counts.values()) / len(patch_counts)
            if patch_counts
            else 0,
            "avg_wsis_per_patient": sum(len(wsis) for wsis in wsi_per_patient.values())
            / len(wsi_per_patient)
            if wsi_per_patient
            else 0,
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        """Utility method for moving all elements of the batch to a device"""
        batch["volume"] = batch["volume"].to(device)
        batch["label"] = batch["label"].to(device)


def sanity_check_dataset():
    # Test both sampling strategies
    for strategy in ["random", "consecutive"]:
        print(f"\nTesting {strategy} sampling strategy:")

        # Instantiate the dataset
        dataset = UnimodalWSIDataset3D(
            split="all",
            dataset_path="data/processed/processed_CPTAC_PDA_71_3D",
            patches_per_wsi=100,
            sampling_strategy=strategy,
        )

        # Check stats of dataset
        print("\nDataset stats:")
        stats = dataset.stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Check random items
        print("\nChecking random items:")
        for i in range(2):
            min_idx = 0
            max_idx = len(dataset) - 1
            item = dataset[randint(min_idx, max_idx)]
            print(f"Item {i}:")
            print(f"  Patient ID: {item['patient_id']}")
            print(f"  Volume shape: {item['volume'].shape}")
            print(f"  Label: {item['label']}")
            print(f"  Slide folder: {item['slide']}")

        # Test DataLoader
        print("\nTesting DataLoader:")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        print(f"Batch volume shape: {batch['volume'].shape}")


if __name__ == "__main__":
    sanity_check_dataset()
