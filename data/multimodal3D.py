import concurrent.futures
import os
import random
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchstain
from torchvision import transforms

class MultimodalCTWSIDataset(Dataset):
    """
    Dataset class for paired CT and WSI data, handling missing modalities
        at both dataset and sampling level
    """

    num_classes = 3
    map_classes = {"G1": 0, "G2": 1, "G3": 2}

    def __init__(
        self,
        split: str,
        dataset_path: str,
        patches_per_wsi: int = 66,
        sampling_strategy: str = "consecutive",
        missing_modality_prob: float = 0.0,  # Additional random masking probability
        vol_std_method: str = None,
        require_both_modalities: bool = False,  # Whether to only include patients
        # with both modalities
        pairing_mode: str = None,  # 'all_combinations, 'one_to_one', 'fixed_count'
        pairs_per_patient: int = None,  # For fixed_count_mode
        allow_repeats: bool = False,  # For fixed_count mode
        downsample: bool = False,
        histo_normalization: str = None,
        std_target: str = None,
        augment_minority: bool = False,
        augmentation_factor: int = 2,
    ):
        super().__init__()
        # assert split in ["train", "val", "overfit", "all"]
        assert sampling_strategy in [
            "random",
            "consecutive",
            "consecutive-fixed",
            "consecutive-seq",
        ]
        assert pairing_mode in ["all_combinations", "one_to_one", "fixed_count"]
        assert 0 <= missing_modality_prob <= 1

        self.dataset_path = dataset_path
        self.patches_per_wsi = patches_per_wsi
        self.sampling_strategy = sampling_strategy
        self.missing_modality_prob = missing_modality_prob
        self.vol_std_method = vol_std_method
        self.require_both_modalities = require_both_modalities
        self.augment_minority = augment_minority  # Flag to enable minority class augmentation
        self.augmentation_factor = augmentation_factor  # How many times to augment the minority class
        
        self.pairing_mode = (
            pairing_mode  # 'all_combinations', 'one_to_one', or 'fixed_count'
        )
        self.pairs_per_patient = pairs_per_patient  # For fixed_count mode
        self.allow_repeats = allow_repeats  # For fixed_count mode
        self.downsample = downsample
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
        self.histo_normalization = histo_normalization
        self.std_target = std_target
        if self.histo_normalization == "macenko" :
            if not self.std_target:
                raise ValueError(
                    "If 'histo_normalization' is set to macenko, 'std_target' cannot be None"
                )
            self.target = cv2.cvtColor(cv2.imread(self.std_target), cv2.COLOR_BGR2RGB)
            self.macenko_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
            self.macenko_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x*255)
            ])
            self.macenko_normalizer.fit(self.macenko_transform(self.target))
            
            
        
        # Initialize data structures
        # Will store CT and WSI paths per patient
        self.patient_data = {}
        # Will store all valid combinations
        self.samples = []
        self.classfreq = {"G1": 0, "G2": 0, "G3": 0}
        self.modality_stats = {"ct_only": 0, "wsi_only": 0, "both": 0}

        # if self.histo_normalization is not None:
        #    if self.histo_normalization == "vahadane":
        #        self.normalizer = staintools.StainNormalizer(method="vahadane")
        #        self.normalizer.fit(self.std_target)

        # Load split file
        self._load_split(split)

        # Calculate maximum CT depth for standardization
        self.max_ct_depth = self._calculate_max_ct_depth()
        # Apply augmentation for minority class if enabled
        if self.augment_minority and "G1" in self.classfreq and self.classfreq["G1"] > 0:
            self._augment_minority_class()
        # Define augmentation transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])

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
    def _augment_minority_class(self):
        """Augment the minority class (G1) samples"""
        # Get all G1 samples
        g1_samples = [sample for sample in self.samples if self.labels[sample["patient_id"]] == "G1"]
        
        # Create augmented versions of G1 samples
        augmented_samples = []
        for _ in range(self.augmentation_factor - 1):  # -1 because we already have one copy
            for sample in g1_samples:
                augmented_sample = sample.copy()
                augmented_sample["augmented"] = True
                augmented_samples.append(augmented_sample)
        
        # Add augmented samples to the dataset
        self.samples.extend(augmented_samples)
        
        # Update class frequency
        self.classfreq["G1"] *= self.augmentation_factor
        
        print(f"Added {len(augmented_samples)} augmented G1 samples. New G1 frequency: {self.classfreq['G1']}")

    def _load_split(self, split):
        """Load and organize all CT and WSI data for the given split

        Supports different pairing modes:
        - 'all_combinations': Creates all possible CT-WSI pairs
        - 'one_to_one': Creates random 1:1 pairs
        - 'fixed_count': Creates fixed number of pairs per patient
        """

        class_counts = [0, 0, 0]  # G1, G2, G3
        # First pass: count maximum possible pairs per patient
        max_pairs_possible = float("inf")
        if self.pairing_mode == "fixed_count" or self.downsample:
            for row in open(f"{os.path.join(self.dataset_path, split)}.txt"):
                patient_id = row.strip()

                ct_path = os.path.join(self.dataset_path, "CT", patient_id)
                ct_scans = []
                if os.path.exists(ct_path):
                    ct_scans = [f for f in os.listdir(ct_path)]

                wsi_path = os.path.join(self.dataset_path, "WSI")
                wsi_folders = [
                    f
                    for f in os.listdir(wsi_path)
                    if patient_id in f and os.path.isdir(os.path.join(wsi_path, f))
                ]

                patient_max_pairs = self._get_max_pairs_for_patient(
                    ct_scans, wsi_folders, self.allow_repeats
                )

                if patient_max_pairs > 0:  # Only update if patient has both modalities

                    max_pairs_possible = min(max_pairs_possible, patient_max_pairs)
                    class_counts[
                        self.map_classes[self.labels[patient_id]]
                    ] += patient_max_pairs

        downsample_trunc = min(class_counts)  # downsampling threshold

        # Use provided pairs_per_patient or calculated maximum
        n_pairs = (
            self.pairs_per_patient
            if self.pairs_per_patient is not None
            else max_pairs_possible
        )

        # Main loading loop
        with open(f"{os.path.join(self.dataset_path, split)}.txt") as split_file:
            for row in split_file:
                patient_id = row.strip()
                # Don't add patients to class if class already downsampled
                if (
                    self.downsample
                    and self.classfreq[self.labels[patient_id]] >= downsample_trunc
                ):
                    continue
                # Find all CT scans for this patient
                ct_path = os.path.join(self.dataset_path, "CT", patient_id)
                ct_scans = []
                if os.path.exists(ct_path):
                    ct_scans = [f for f in os.listdir(ct_path)]

                # Find all WSI folders for this patient
                wsi_path = os.path.join(self.dataset_path, "WSI")
                wsi_folders = [
                    f
                    for f in os.listdir(wsi_path)
                    if patient_id in f and os.path.isdir(os.path.join(wsi_path, f))
                ]

                # Skip patient if we require both modalities and they don't have them
                if self.require_both_modalities and (not ct_scans or not wsi_folders):
                    continue

                # Store available data for this patient
                self.patient_data[patient_id] = {
                    "ct_scans": ct_scans,
                    "wsi_folders": wsi_folders,
                }

                # Update modality statistics
                if ct_scans and wsi_folders:
                    self.modality_stats["both"] += 1
                elif ct_scans:
                    self.modality_stats["ct_only"] += 1
                else:
                    self.modality_stats["wsi_only"] += 1

                ub = downsample_trunc - self.classfreq[self.labels[patient_id]]
                cnt = 0
                # Generate samples based on available data and pairing mode
                if ct_scans and wsi_folders:
                    if self.pairing_mode == "fixed_count":
                        # Generate fixed number of pairs
                        pairs = self._get_fixed_pairs(
                            ct_scans, wsi_folders, n_pairs, self.allow_repeats
                        )

                        for ct_scan, wsi_folder in pairs:
                            if cnt >= ub and self.downsample:
                                break
                            cnt += 1
                            self.samples.append(
                                {
                                    "patient_id": patient_id,
                                    "ct_path": os.path.join("CT", patient_id, ct_scan),
                                    "wsi_folder": os.path.join("WSI", wsi_folder),
                                    "base_modality_mask": [1, 1],
                                    "augmented": False,
                                }
                            )

                    elif self.pairing_mode == "one_to_one":
                        # Original one_to_one logic
                        num_pairs = min(len(ct_scans), len(wsi_folders))
                        shuffled_ct = ct_scans.copy()
                        shuffled_wsi = wsi_folders.copy()
                        random.shuffle(shuffled_ct)
                        random.shuffle(shuffled_wsi)

                        for ct_scan, wsi_folder in zip(
                            shuffled_ct[:num_pairs], shuffled_wsi[:num_pairs]
                        ):
                            if cnt >= ub and self.downsample:
                                break
                            cnt += 1
                            self.samples.append(
                                {
                                    "patient_id": patient_id,
                                    "ct_path": os.path.join("CT", patient_id, ct_scan),
                                    "wsi_folder": os.path.join("WSI", wsi_folder),
                                    "base_modality_mask": [1, 1],
                                    "augmented": False,
                                }
                            )

                    else:  # 'all_combinations' mode
                        for ct_scan, wsi_folder in product(ct_scans, wsi_folders):
                            if cnt >= ub and self.downsample:
                                break
                            cnt += 1
                            self.samples.append(
                                {
                                    "patient_id": patient_id,
                                    "ct_path": os.path.join("CT", patient_id, ct_scan),
                                    "wsi_folder": os.path.join("WSI", wsi_folder),
                                    "base_modality_mask": [1, 1],
                                    "augmented": False,
                                }
                            )

                elif ct_scans:
                    for ct_scan in ct_scans:
                        if cnt >= ub and self.downsample:
                            break
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": os.path.join("CT", patient_id, ct_scan),
                                "wsi_folder": None,
                                "base_modality_mask": [1, 0],
                                "augmented": False
                            }
                        )

                elif wsi_folders:
                    for wsi_folder in wsi_folders:
                        if cnt >= ub and self.downsample:
                            break
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": None,
                                "wsi_folder": os.path.join("WSI", wsi_folder),
                                "base_modality_mask": [0, 1],
                                "augmented": False
                            }
                        )

                self.classfreq[self.labels[patient_id]] += len(self.samples) - sum(
                    self.classfreq.values()
                )

    def _calculate_max_ct_depth(self):
        """Calculate maximum CT depth across all samples"""
        max_depth = 0
        for pair in self.samples:
            ct_path = os.path.join(self.dataset_path, pair["ct_path"])
            if os.path.exists(ct_path):
                depth = len(np.load(ct_path))
                max_depth = max(max_depth, depth)
        return max_depth

    def _load_ct_volume(self, ct_path, augment = False):
        """Load and standardize CT volume"""
        volume = np.load(os.path.join(self.dataset_path, ct_path))
        if self.vol_std_method == "padding":
            # Pad to max_depth if necessary
            if volume.shape[0] < self.max_ct_depth:
                pad_before = (self.max_ct_depth - volume.shape[0]) // 2
                pad_after = self.max_ct_depth - volume.shape[0] - pad_before
                volume = np.pad(
                    volume,
                    ((pad_before, pad_after), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        
        # Apply CT-specific augmentations if this is an augmented sample
        if augment:
            # Add random noise
            noise = np.random.normal(0, 0.03, volume.shape).astype(volume.dtype)
            volume = volume + noise
            
            # Random intensity shift
            intensity_shift = np.random.uniform(-0.1, 0.1)
            volume = volume + intensity_shift
            
            # Random contrast adjustment
            contrast_factor = np.random.uniform(0.9, 1.1)
            volume = (volume - volume.mean()) * contrast_factor + volume.mean()
            
            # Ensure values are within valid range
            volume = np.clip(volume, volume.min(), volume.max())
        return volume

    def _load_wsi_volume(self, wsi_folder, augment=False):
        """Load WSI patches into a volume"""
        patches = []
        patch_files = sorted(
            f
            for f in os.listdir(os.path.join(self.dataset_path, wsi_folder))
            if f.endswith((".png", ".jpg", ".jpeg"))
        )

        # Sample patches according to strategy
        if len(patch_files) <= self.patches_per_wsi:
            selected_patches = patch_files * (self.patches_per_wsi // len(patch_files))
            selected_patches += random.sample(
                patch_files, self.patches_per_wsi % len(patch_files)
            )
        elif self.sampling_strategy == "random":
            selected_patches = random.sample(patch_files, self.patches_per_wsi)
        elif self.sampling_strategy == "consecutive":  # consecutive
            start_idx = random.randint(0, len(patch_files) - self.patches_per_wsi)
            selected_patches = patch_files[
                start_idx : start_idx + self.patches_per_wsi  # noqa E203
            ]
        elif self.sampling_strategy == "consecutive-fixed":  # Consecutive - fixed
            start_idx = round((len(patch_files) - self.patches_per_wsi) / 2)
            selected_patches = patch_files[
                start_idx : start_idx + self.patches_per_wsi  # noqa E203
            ]
        else:  # Consecutive for sequences "consecutive-seq"
            start_idx = 0
            selected_patches = patch_files[
                start_idx : start_idx + self.patches_per_wsi  # noqa E203
            ]
        # Load selected patches
        # for patch_file in selected_patches:
        #     patch_path = os.path.join(self.dataset_path, wsi_folder, patch_file)
        #     img = cv2.imread(patch_path)
        #     img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     patches.append(img_np)

        # Parallel load
        # Create the full file paths
        patch_paths = [
            os.path.join(self.dataset_path, wsi_folder, patch_file)
            for patch_file in selected_patches
        ]

        # Use ThreadPoolExecutor to process the patches in parallel
        augments = [augment for i in range(len(patch_paths))] 
        with concurrent.futures.ThreadPoolExecutor() as executor:
            patches = list(executor.map(self._process_patch, patch_paths,augments))
        try:
            return np.stack(patches, axis=0).transpose(3, 0, 1, 2)
        except ValueError as e:
            print("patches:",len(patches))
            print("patch_paths:",len(patch_paths))

    def _process_patch(self, patch_path, augment=False):
        img = cv2.imread(patch_path)
        img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_np
        if self.histo_normalization is not None:
            # if self.histo_normalization == "vahadane":
            #     img = staintools.LuminosityStandardizer.standardize(img_np)
            #     img_normalized = self.normalizer.transform(img)
            if self.histo_normalization == "imagenet":
                imagenet_mean = np.array(
                    [0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32
                )
                imagenet_std = np.array(
                    [0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32
                )
                img_normalized = (img_np - imagenet_mean) / imagenet_std
            if self.histo_normalization == "macenko":
                if img_np is None or img_np.size == 0:
                    print("Warning: Empty patch encountered. Skipping...")
                    return None  # or a default tensor
                norm, _, _ = self.macenko_normalizer.normalize(I=self.macenko_transform(img_np), stains=True)
                    
                
                img_normalized = norm.byte().numpy()
                                
                #norm, _,_ = self.macenko_normalizer.normalize(I=self.macenko_transform(img_np),stains=True)
        # assert isinstance(img, np.ndarray), "img is not a numpy array!"
        # Apply augmentations if this is an augmented sample
        if augment:
            # Convert to PIL Image for torchvision transforms
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float() / 255.0
            
            # Apply random transformation
            img_tensor = self.augmentation_transforms(img_tensor)
            
            # Convert back to numpy array
            img_normalized = (img_tensor * 255.0).permute(1, 2, 0).byte().numpy()
        assert img_normalized is not None, f"{patch_path} is None!"
        return img_normalized

    def _get_empty_ct_volume(self):
        """Return empty CT volume of correct shape"""
        return np.zeros((self.max_ct_depth, 224, 224))

    def _get_empty_wsi_volume(self):
        """Return empty WSI volume of correct shape"""
        return np.zeros((3, self.patches_per_wsi, 224, 224))

    def __getitem__(self, index):
        """
        Returns:
            dict: {
                'patient_id': str,
                'ct_volume': numpy array or zeros if missing,
                'wsi_volume': tensor or zeros if missing,
                'label': int,
                'modality_mask': tensor indicating present modalities [CT, WSI],
                'base_modality_mask': tensor indicating modalities available in dataset
            }
        """

        sample = self.samples[index]
        patient_id = sample["patient_id"]
        base_mask = sample["base_modality_mask"]
        is_augmented = sample["augmented"]
        # Apply additional random masking only to available modalities
        final_mask = base_mask.copy()
        if self.missing_modality_prob > 0:
            for i in range(2):
                if base_mask[i] == 1 and random.random() < self.missing_modality_prob:
                    final_mask[i] = 0

            # Ensure at least one modality remains if it was originally available
            if sum(final_mask) == 0 and sum(base_mask) > 0:
                # Randomly choose one of the originally available modalities
                available_indices = [i for i in range(2) if base_mask[i] == 1]
                chosen_idx = random.choice(available_indices)
                final_mask[chosen_idx] = 1

        # Load volumes based on final mask
        ct_volume = (
            self._load_ct_volume(sample["ct_path"], augment=is_augmented)
            if final_mask[0] and sample["ct_path"]
            else self._get_empty_ct_volume()
        )

        wsi_volume = (
            self._load_wsi_volume(sample["wsi_folder"], augment=is_augmented)
            if final_mask[1] and sample["wsi_folder"]
            else self._get_empty_wsi_volume()
        )

        return {
            "patient_id": patient_id,
            "ct_volume": torch.from_numpy(ct_volume).float(),
            "wsi_volume": torch.from_numpy(wsi_volume).float(),
            "label": torch.tensor(
                self.map_classes[self.labels[patient_id]], dtype=torch.long
            ),
            "modality_mask": torch.tensor(final_mask, dtype=torch.float32),
            "base_modality_mask": torch.tensor(base_mask, dtype=torch.float32),
            "augmented": torch.tensor(is_augmented, dtype=torch.bool)
        }

    def __len__(self):
        return len(self.samples)

    def stats(self):
        """Return dataset statistics"""
        augmented_count = sum(1 for sample in self.samples if sample["augmented"])
        return {
            "total_samples": len(self.samples),
            "total_patients": len(self.patient_data),
            "class_frequency": self.classfreq,
            "augmented_samples": augmented_count,
            "modality_availability": self.modality_stats,
            "max_ct_depth": self.max_ct_depth,
            "patches_per_wsi": self.patches_per_wsi,
            "sampling_strategy": self.sampling_strategy,
            "missing_modality_prob": self.missing_modality_prob,
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        """Move all elements of the batch to device"""
        batch["ct_volume"] = batch["ct_volume"].to(device)
        batch["wsi_volume"] = batch["wsi_volume"].to(device)
        batch["label"] = batch["label"].to(device)
        batch["modality_mask"] = batch["modality_mask"].to(device)
        if "augmented" in batch:
            batch["augmented"] = batch["augmented"].to(device)

class MultimodalBinaryCTWSIDataset(MultimodalCTWSIDataset):
    num_classes = 3
    map_classes = {"G2": 0, "G3": 1}
    
    def __init__(
        self,
        split: str,
        dataset_path: str,
        patches_per_wsi: int = 66,
        sampling_strategy: str = "consecutive",
        missing_modality_prob: float = 0.0,  # Additional random masking probability
        vol_std_method: str = None,
        require_both_modalities: bool = False,  # Whether to only include patients
        # with both modalities
        pairing_mode: str = None,  # 'all_combinations, 'one_to_one', 'fixed_count'
        pairs_per_patient: int = None,  # For fixed_count_mode
        allow_repeats: bool = False,  # For fixed_count mode
        downsample: bool = False,
        histo_normalization: str = None,
        std_target: str = None,
    ):
        # assert split in ["train", "val", "overfit", "all"]
        assert sampling_strategy in [
            "random",
            "consecutive",
            "consecutive-fixed",
            "consecutive-seq",
        ]
        assert pairing_mode in ["all_combinations", "one_to_one", "fixed_count"]
        assert 0 <= missing_modality_prob <= 1

        self.dataset_path = dataset_path
        self.patches_per_wsi = patches_per_wsi
        self.sampling_strategy = sampling_strategy
        self.missing_modality_prob = missing_modality_prob
        self.vol_std_method = vol_std_method
        self.require_both_modalities = require_both_modalities

        self.pairing_mode = (
            pairing_mode  # 'all_combinations', 'one_to_one', or 'fixed_count'
        )
        self.pairs_per_patient = pairs_per_patient  # For fixed_count mode
        self.allow_repeats = allow_repeats  # For fixed_count mode
        self.downsample = downsample
        # Load labels
        self.labels = {
            k.strip(): v.strip()
            for k, v in (
                line.split(",")
                for line in Path(f'{os.path.join(dataset_path, "labelsG23.txt")}')
                .read_text()
                .splitlines()
            )
        }
        self.histo_normalization = histo_normalization
        self.std_target = std_target
        if self.histo_normalization == "macenko" :
            if not self.std_target:
                raise ValueError(
                    "If 'histo_normalization' is set to macenko, 'std_target' cannot be None"
                )
            self.target = cv2.cvtColor(cv2.imread(self.std_target), cv2.COLOR_BGR2RGB)
            self.macenko_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
            self.macenko_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x*255)
            ])
            self.macenko_normalizer.fit(self.macenko_transform(self.target))
            
            
        
        # Initialize data structures
        # Will store CT and WSI paths per patient
        self.patient_data = {}
        # Will store all valid combinations
        self.samples = []
        self.classfreq = {"G2": 0, "G3": 1}
        self.modality_stats = {"ct_only": 0, "wsi_only": 0, "both": 0}

        # if self.histo_normalization is not None:
        #    if self.histo_normalization == "vahadane":
        #        self.normalizer = staintools.StainNormalizer(method="vahadane")
        #        self.normalizer.fit(self.std_target)

        # Load split file
        self._load_split(split)

        # Calculate maximum CT depth for standardization
        self.max_ct_depth = self._calculate_max_ct_depth()
# ---------------------------------------------------------------------


def test_multimodal_dataset():
    # Initialize dataset
    dataset = MultimodalCTWSIDataset(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
        patches_per_wsi=66,
        sampling_strategy="consecutive-fixed",
        missing_modality_prob=0.0,  # 20% chance of each modality being missing
        require_both_modalities=True,
        pairing_mode="all_combinations",
        allow_repeats=False,
        pairs_per_patient=None,
        downsample=False,
        histo_normalization="macenko",
        std_target="data/processed/processed_CPTAC_PDA_71_3D/WSI/C3L-00625-25/96.png",
        augment_minority= True,
        augmentation_factor= 2
        )

    # Print dataset stats
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
        print(f"  CT Volume Shape: {sample['ct_volume'].shape}")
        print(f"  WSI Volume Shape: {sample['wsi_volume'].shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Modality Mask: {sample['modality_mask'].tolist()}")

    # Test DataLoader
    print("\nTesting DataLoader:")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))

    print("Batch shapes:")
    print(f"  CT volumes: {batch['ct_volume'].shape}")
    print(f"  WSI volumes: {batch['wsi_volume'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  Modality masks: {batch['modality_mask'].shape}")

def test_binary_multimodal_dataset():
    # Initialize dataset
    dataset = MultimodalBinaryCTWSIDataset(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
        patches_per_wsi=66,
        sampling_strategy="consecutive-fixed",
        missing_modality_prob=0.0,  # 20% chance of each modality being missing
        require_both_modalities=True,
        pairing_mode="all_combinations",
        allow_repeats=False,
        pairs_per_patient=None,
        downsample=False,
        histo_normalization=None,
        std_target="data/processed/processed_CPTAC_PDA_71_3D/WSI/C3L-00625-25/96.png",
    )

    # Print dataset stats
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
        print(f"  CT Volume Shape: {sample['ct_volume'].shape}")
        print(f"  WSI Volume Shape: {sample['wsi_volume'].shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Modality Mask: {sample['modality_mask'].tolist()}")

    # Test DataLoader
    print("\nTesting DataLoader:")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))

    print("Batch shapes:")
    print(f"  CT volumes: {batch['ct_volume'].shape}")
    print(f"  WSI volumes: {batch['wsi_volume'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  Modality masks: {batch['modality_mask'].shape}")
if __name__ == "__main__":
    test_multimodal_dataset()
