# TODO: Voglio fare un Dataset che prende in ingresso il path per le feature CT,
# il path per le feature WSI, il path per le labels (che contiene anche gli split)
# e il numero del fold (tra 0 e 4), e una stringa "train" o "test", in base al numero
# del fold carica in memoria i sample (che saranno coppie di feature WSI e associata CT,
# senza ripetere le WSI, ma ripetendo le CT se necessario)
import concurrent.futures
import os
import random
from itertools import product
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torchstain
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class MultimodalCTWSIDatasetSurv(Dataset):
    """
    Dataset class for paired CT and WSI data, handling missing modalities
        at both dataset and sampling level
    """

    def __init__(
        self,
        fold: int,
        split: str,  # either "train" or "test"
        ct_path: str,
        wsi_path: str,
        labels_splits_path: str,
        missing_modality_prob: float = 0.0,  # Additional random masking probability
        missing_modality: str = "both",
        require_both_modalities: bool = False,  # Whether to only include patients
        # with both modalities
        pairing_mode: str = None,  # 'all_combinations, 'one_to_one', 'fixed_count'
        pairs_per_patient: int = None,  # For fixed_count_mode
        allow_repeats: bool = False,  # For fixed_count mode
    ):
        super().__init__()
        # assert split in ["train", "val", "overfit", "all"]
        assert split in ["train", "test"]
        assert pairing_mode in ["all_combinations", "one_to_one", "fixed_count"]
        assert 0 <= missing_modality_prob <= 1
        assert missing_modality in ["ct", "wsi", "both"]
        self.missing_modality = missing_modality
        self.fold = fold
        self.split = split
        self.ct_path = ct_path
        self.wsi_path = wsi_path
        self.missing_modality_prob = missing_modality_prob
        self.require_both_modalities = require_both_modalities
        self.labels_splits_path = labels_splits_path
        self.pairing_mode = (
            pairing_mode  # 'all_combinations', 'one_to_one', or 'fixed_count'
        )
        self.pairs_per_patient = pairs_per_patient  # For fixed_count mode
        self.allow_repeats = allow_repeats  # For fixed_count mode
        labels_splits = pd.read_csv(labels_splits_path, sep="\t")
        labels_splits = labels_splits[labels_splits[f"fold_{self.fold}"] == self.split]
        self.labels_splits = labels_splits[
            ["case_id", "OS_days", "OS_event"]
        ].drop_duplicates("case_id")

        # Initialize data structures
        # Will store CT and WSI paths per patient
        self.patient_data = {}
        # Will store all valid combinations
        self.samples = []
        self.modality_stats = {"ct_only": 0, "wsi_only": 0, "both": 0}

        # Load split file
        self._load_split()

    def _get_max_pairs_for_patient(self, ct_scans, wsi_folders, allow_repeats):
        """
        Calculate maximum possible pairs for a patient based on available data and pairing mode.

        Args:
            ct_scans (list): List of CT scan files
            wsi_folders (list): List of WSI folder names
            allow_repeats (bool): If True, return all possible combinations count
                                If False, return maximum unique pairs count

        Returns:
            int: Maximum number of possible pairs
        """
        if not ct_scans or not wsi_folders:
            return 0

        if allow_repeats:
            return len(ct_scans) * len(wsi_folders)
        else:  # one-to-one
            return min(len(ct_scans), len(wsi_folders))

    def _get_fixed_pairs(self, ct_scans, wsi_folders, n_pairs, allow_repeats=True):
        """
        Generate fixed number of pairs between CT scans and WSI folders.

        Args:
            ct_scans (list): List of CT scan files
            wsi_folders (list): List of WSI folder names
            n_pairs (int): Number of pairs to generate
            allow_repeats (bool): If True, allows repeating elements to reach n_pairs
                                If False, limits pairs to minimum unique combinations

        Returns:
            list: List of (ct_scan, wsi_folder) pairs
        """
        max_unique_pairs = min(len(ct_scans), len(wsi_folders))

        if not allow_repeats:
            n_pairs = min(n_pairs, max_unique_pairs)

        # Generate initial unique pairs
        shuffled_ct = ct_scans.copy()
        shuffled_wsi = wsi_folders.copy()
        random.shuffle(shuffled_ct)
        random.shuffle(shuffled_wsi)

        pairs = list(
            zip(shuffled_ct[:max_unique_pairs], shuffled_wsi[:max_unique_pairs])
        )

        if n_pairs <= len(pairs):
            # Downsample if needed
            random.shuffle(pairs)
            return pairs[:n_pairs]

        if not allow_repeats:
            return pairs

        # Need to generate additional pairs with repeats
        while len(pairs) < n_pairs:
            ct_scan = random.choice(ct_scans)
            wsi_folder = random.choice(wsi_folders)
            pairs.append((ct_scan, wsi_folder))

        return pairs

    def _load_split(self):
        """Load and organize all CT and WSI data for the given split

        Supports different pairing modes:
        - 'all_combinations': Creates all possible CT-WSI pairs
        - 'one_to_one': Creates random 1:1 pairs
        - 'fixed_count': Creates fixed number of pairs per patient
        """

        # First pass: count maximum possible pairs per patient
        max_pairs_possible = float("inf")
        if self.pairing_mode == "fixed_count":
            for patient_id in self.labels_splits["case_id"].values:

                ct_path = os.path.join(self.ct_path, patient_id)
                ct_features = []
                if os.path.exists(ct_path):
                    ct_features = [f for f in os.listdir(ct_path)]

                wsi_path = os.path.join(self.wsi_path)
                wsi_features = [
                    f
                    for f in os.listdir(wsi_path)
                    if patient_id in f and os.path.isdir(os.path.join(wsi_path, f))
                ]

                patient_max_pairs = self._get_max_pairs_for_patient(
                    ct_features, wsi_features, self.allow_repeats
                )

                if patient_max_pairs > 0:  # Only update if patient has both modalities

                    max_pairs_possible = min(max_pairs_possible, patient_max_pairs)

        # Use provided pairs_per_patient or calculated maximum
        n_pairs = (
            self.pairs_per_patient
            if self.pairs_per_patient is not None
            else max_pairs_possible
        )

        # Main loading loop
        for row in self.labels_splits["case_id"].values:
            patient_id = row.strip()
            # Find all CT scans for this patient
            ct_path = os.path.join(self.ct_path, patient_id)
            ct_features = []
            if os.path.exists(ct_path):
                ct_features = [f for f in os.listdir(ct_path)]

            # Find all WSI .h5 files for this patient
            wsi_path = self.wsi_path
            wsi_features = [f for f in os.listdir(wsi_path) if patient_id in f]

            # Skip patient if we require both modalities and they don't have them
            if self.require_both_modalities and (not ct_features or not wsi_features):
                continue

            # Store available data for this patient
            self.patient_data[patient_id] = {
                "ct_features": ct_features,
                "wsi_features": wsi_features,
            }

            # Update modality statistics
            if ct_features and wsi_features:
                self.modality_stats["both"] += 1
            elif ct_features:
                self.modality_stats["ct_only"] += 1
            else:
                self.modality_stats["wsi_only"] += 1

            cnt = 0
            # Generate samples based on available data and pairing mode
            if ct_features and wsi_features:
                if self.pairing_mode == "fixed_count":
                    # Generate fixed number of pairs
                    pairs = self._get_fixed_pairs(
                        ct_features, wsi_features, n_pairs, self.allow_repeats
                    )

                    for ct_feature, wsi_feature in pairs:
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": os.path.join(
                                    self.ct_path, patient_id, ct_feature
                                ),
                                "wsi_folder": os.path.join(self.wsi_path, wsi_feature),
                                "base_modality_mask": [1, 1],
                            }
                        )

                elif self.pairing_mode == "one_to_one":
                    # Original one_to_one logic
                    num_pairs = min(len(ct_features), len(wsi_features))
                    shuffled_ct = ct_features.copy()
                    shuffled_wsi = wsi_features.copy()
                    random.shuffle(shuffled_ct)
                    random.shuffle(shuffled_wsi)

                    for ct_feature, wsi_feature in zip(
                        shuffled_ct[:num_pairs], shuffled_wsi[:num_pairs]
                    ):
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": os.path.join(
                                    self.ct_path, patient_id, ct_feature
                                ),
                                "wsi_feature": os.path.join(self.wsi_path, wsi_feature),
                                "base_modality_mask": [1, 1],
                            }
                        )

                else:  # 'all_combinations' mode
                    for ct_feature, wsi_feature in product(ct_features, wsi_features):
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": os.path.join(
                                    self.ct_path, patient_id, ct_feature
                                ),
                                "wsi_feature": os.path.join(self.wsi_path, wsi_feature),
                                "base_modality_mask": [1, 1],
                            }
                        )

            elif ct_features:
                for ct_feature in ct_features:
                    cnt += 1
                    self.samples.append(
                        {
                            "patient_id": patient_id,
                            "ct_path": os.path.join(
                                self.ct_path, patient_id, ct_feature
                            ),
                            "wsi_feature": None,
                            "base_modality_mask": [1, 0],
                        }
                    )

            elif wsi_features:
                for wsi_feature in wsi_features:
                    cnt += 1
                    self.samples.append(
                        {
                            "patient_id": patient_id,
                            "ct_path": None,
                            "wsi_feature": os.path.join(self.wsi_path, wsi_feature),
                            "base_modality_mask": [0, 1],
                        }
                    )

    def _load_ct_feature(self, ct_path):
        """Load and standardize CT feature"""
        volume = np.load(ct_path)
        return volume

    def _load_wsi_feature(self, wsi_path):
        """Load WSI feature"""
        feature = None
        with h5py.File(wsi_path, "r") as f:
            feature = np.array(f["features"][:])

        return feature

    def _get_empty_ct_feature(self):
        """Return empty CT feature of correct shape"""
        return np.zeros((66, 1024))

    def _get_empty_wsi_feature(self):
        """Return empty WSI feature of correct shape"""
        return np.zeros((768,))

    def __getitem__(self, index):
        """
        Returns:
            dict: {
                'patient_id': str,
                'ct_feature': numpy array or zeros if missing,
                'wsi_feature': tensor or zeros if missing,
                'label': int,
                'modality_mask': tensor indicating present modalities [CT, WSI],
                'base_modality_mask': tensor indicating modalities available in dataset
            }
        """

        sample = self.samples[index]
        patient_id = sample["patient_id"]
        base_mask = sample["base_modality_mask"]

        # Apply additional random masking only to available modalities
        final_mask = base_mask.copy()
        if self.missing_modality_prob > 0:
            if self.missing_modality == "both":
                for i in range(2):
                    if (
                        base_mask[i] == 1
                        and random.random() < self.missing_modality_prob
                    ):
                        final_mask[i] = 0
            elif self.missing_modality == "ct":
                if base_mask[0] == 1 and random.random() < self.missing_modality_prob:
                    final_mask[0] = 0
            elif self.missing_modality == "wsi":
                if base_mask[1] == 1 and random.random() < self.missing_modality_prob:
                    final_mask[1] = 0

            # Ensure at least one modality remains if it was originally available
            if sum(final_mask) == 0 and sum(base_mask) > 0:
                # Randomly choose one of the originally available modalities
                available_indices = [i for i in range(2) if base_mask[i] == 1]
                chosen_idx = random.choice(available_indices)
                final_mask[chosen_idx] = 1

        # Load features based on final mask
        ct_feature = (
            self._load_ct_feature(sample["ct_path"])
            if final_mask[0] and sample["ct_path"]
            else self._get_empty_ct_feature()
        )

        wsi_feature = (
            self._load_wsi_feature(sample["wsi_feature"])
            if final_mask[1] and sample["wsi_feature"]
            else self._get_empty_wsi_feature()
        )

        return {
            "patient_id": patient_id,
            "ct_feature": torch.from_numpy(ct_feature).float(),
            "wsi_feature": torch.from_numpy(wsi_feature).float(),
            "survtime": torch.tensor(
                self.labels_splits[self.labels_splits["case_id"] == patient_id][
                    "OS_days"
                ].iloc[0],
                dtype=torch.long,
            ),
            "censor": ~torch.tensor(
                self.labels_splits[self.labels_splits["case_id"] == patient_id][
                    "OS_event"
                ].iloc[0],
                dtype=torch.bool,
            ),
            "modality_mask": torch.tensor(final_mask, dtype=torch.float32),
            "base_modality_mask": torch.tensor(base_mask, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.samples)

    def stats(self):
        """Return dataset statistics"""
        return {
            "total_samples": len(self.samples),
            "total_patients": len(self.patient_data),
            "modality_availability": self.modality_stats,
            "missing_modality_prob": self.missing_modality_prob,
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        """Move all elements of the batch to device"""
        batch["ct_feature"] = batch["ct_feature"].to(device)
        batch["wsi_feature"] = batch["wsi_feature"].to(device)
        batch["survtime"] = batch["survtime"].to(device)
        batch["censor"] = batch["censor"].to(device)
        batch["modality_mask"] = batch["modality_mask"].to(device)


# ---------------------------------------------------------------------


def test_multimodal_dataset():
    # Initialize dataset
    dataset = MultimodalCTWSIDatasetSurv(
        fold=0,
        split="train",
        ct_path="../MedImageInsights/embeddings_output_cptacpda_93",
        wsi_path="../../TitanCPTACPDA/20x_512px_0px_overlap/slide_features_titan",
        labels_splits_path="./data/processed/processed_CPTAC_PDA_survival/k=all.tsv",
        missing_modality_prob=0.3,  # 30% chance of each modality being missing
        missing_modality="both",
        require_both_modalities=True,
        pairing_mode="one_to_one",
        allow_repeats=True,
        pairs_per_patient=None,
    )

    print("\nDataset Statistics:")
    stats = dataset.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test a few random samples
    print("\nTesting random samples:")
    for i in range(3):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]

        print(f"\nSample {i + 1}:")
        print(f"  Patient ID: {sample['patient_id']}")
        print(f"  CT Feature Shape: {sample['ct_feature'].shape}")
        print(f"  WSI Feature Shape: {sample['wsi_feature'].shape}")
        print(f"  Survtime: {sample['survtime']}")
        print(f"  Censor: {sample['censor']}")
        print(f"  Modality Mask: {sample['modality_mask'].tolist()}")

    # Test DataLoader
    print("\nTesting DataLoader:")
    dataloader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=True)
    #batch = next(iter(dataloader))
    total_samples = 0
    ct_only = 0
    wsi_only = 0
    both_modalities = 0
    no_modalities = 0
    for batch in tqdm(dataloader, desc = "Analyzing dataset"):
        batch_size = len(batch["patient_id"])
        total_samples += batch_size
        
        modality_masks = batch["modality_mask"]
        print(len(modality_masks) != batch_size)
        for i in range(batch_size):
            mask = modality_masks[i].tolist()
            
            if mask[0] == 1 and mask[1] == 1:
                both_modalities += 1
            elif mask[0] == 1 and mask[1] == 0:
                ct_only += 1
            elif mask[0] == 0 and mask[1] == 1:
                wsi_only += 1
            else:
                no_modalities += 1
        
    # Calculate percentages
    ct_only_pct = (ct_only / total_samples) * 100 if total_samples > 0 else 0
    wsi_only_pct = (wsi_only / total_samples) * 100 if total_samples > 0 else 0
    both_pct = (both_modalities / total_samples) * 100 if total_samples > 0 else 0
    none_pct = (no_modalities / total_samples) * 100 if total_samples > 0 else 0
    
    """Pretty print the modality statistics"""
    print("\n===== MODALITY STATISTICS =====")
    
    # Sample-level statistics
    print("\n----- Sample Level Statistics -----")
    
    print(f"Total samples: {total_samples}")
    print(f"CT only: {ct_only} ({ct_only_pct:.2f}%)")
    print(f"WSI only: {wsi_only} ({wsi_only_pct:.2f}%)")
    print(f"Both modalities: {both_modalities} ({both_pct:.2f}%)")
    
    # print("Batch shapes:")
    # print(f"  CT features: {batch['ct_feature'].shape}")
    # print(f"  WSI features: {batch['wsi_feature'].shape}")
    # print(f"  Survtimes: {batch['survtime'].shape}")
    # print(f"  Censors: {batch['censor'].shape}")
    # print(f"  Modality masks: {batch['modality_mask'].shape}")


if __name__ == "__main__":
    test_multimodal_dataset()
