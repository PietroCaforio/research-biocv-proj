import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "./")  # noqa: E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa: E402
from models.dpe.main_model import madpe_resnet34  # noqa: E402
from training.trainer import MultimodalTrainer  # noqa: E402

SEED = 0


def set_global_seed(seed=SEED):
    """
    Set a global seed for reproducibility across different libraries and random number generators.

    Args:
        seed (int): Seed value to be used
    """
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure PyTorch to make computations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test multimodal model and extract features"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model weights")
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--dataset", type=str, help="Path to test dataset")
    parser.add_argument("--config", type=str, help="Path to config json")
    parser.add_argument(
        "--output_layers",
        nargs="+",
        default=["classification"],
        help="List of output layers to extract features from",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = json.load(f)
    return config


def extract_features(model, data_loader, device, output_layers):
    """
    Extract features from specified output layers for all samples in the dataset

    Args:
        model: The model to extract features from
        data_loader: DataLoader containing the validation dataset
        device: Device to run inference on
        output_layers: List of output layer names to extract features from

    Returns:
        Dictionary mapping layer names to (features, labels) tuples
    """
    model.eval()
    features_dict = {layer: [] for layer in output_layers}
    labels_list = []
    patient_ids = []
    modality_masks = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            # Get data from batch
            ct_volumes = (
                batch["ct_volume"].float().unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)
            )
            wsi_volumes = batch["wsi_volume"].float().to(device)
            labels = batch["label"].to(device)
            modality_mask = batch["modality_mask"].to(device)

            # Store patient IDs if available in the batch
            if "patient_id" in batch:
                patient_ids.extend(batch["patient_id"])

            # Forward pass
            outputs = model(
                ct_volumes,
                wsi_volumes,
                modality_flag=modality_mask,
                output_layers=output_layers,
            )

            # Store features from each requested output layer
            for layer in output_layers:
                if layer in outputs:
                    features_dict[layer].append(outputs[layer].cpu().numpy())

            # Store labels and modality information
            labels_list.append(labels.cpu().numpy())
            modality_masks.append(modality_mask.cpu().numpy())

    # Concatenate features and labels from all batches
    for layer in output_layers:
        if features_dict[layer]:  # Check if any features were collected for this layer
            features_dict[layer] = np.concatenate(features_dict[layer], axis=0)

    labels_array = np.concatenate(labels_list, axis=0)
    modality_masks_array = np.concatenate(modality_masks, axis=0)

    return features_dict, labels_array, modality_masks_array, patient_ids


def save_features(features_dict, labels, modality_masks, patient_ids, output_dir):
    """
    Save extracted features and corresponding labels to files

    Args:
        features_dict: Dictionary of features for each output layer
        labels: Array of class labels
        modality_masks: Array indicating which modalities were present
        patient_ids: List of patient IDs if available
        output_dir: Directory to save features to
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save features from each layer
    for layer_name, features in features_dict.items():
        feature_file = os.path.join(output_dir, f"{layer_name}_features.npy")
        np.save(feature_file, features)
        print(
            f"Saved features from {layer_name} to {feature_file}, shape: {features.shape}"
        )

    # Save labels and modality information
    labels_file = os.path.join(output_dir, "labels.npy")
    np.save(labels_file, labels)

    modality_file = os.path.join(output_dir, "modality_masks.npy")
    np.save(modality_file, modality_masks)

    # Save patient IDs if available
    if patient_ids:
        patient_file = os.path.join(output_dir, "patient_ids.npy")
        np.save(patient_file, np.array(patient_ids))

    # Save metadata for easy reference
    metadata = {
        "num_samples": labels.shape[0],
        "feature_shapes": {
            layer: features.shape for layer, features in features_dict.items()
        },
        "label_distribution": {
            int(label): int(np.sum(labels == label)) for label in np.unique(labels)
        },
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {labels.shape[0]} samples to {output_dir}")


def main():
    set_global_seed(seed=SEED)

    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Save config to output directory
    shutil.copy(args.config, os.path.join(args.out_dir, "config.json"))

    # Set device
    if "gpu" in config and "training" in config and "gpu" in config["training"]:
        torch.cuda.set_device(config["training"]["gpu"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create validation dataset
    val_dataset = MultimodalCTWSIDataset(
        split=config["data"]["val_split"],
        dataset_path=config["data"]["dataset_path"],
        patches_per_wsi=config["data"]["patches_per_wsi"],
        sampling_strategy=config["data"]["sampling_strategy"],
        missing_modality_prob=config["data"]["missing_modality_prob"],
        require_both_modalities=config["data"]["require_both_modalities"],
        pairing_mode=config["data"]["pairing_mode"],
        allow_repeats=config["data"]["allow_repeats"],
        pairs_per_patient=config["data"]["pairs_per_patient"],
        downsample=config["data"]["downsample"],
        histo_normalization=config["data"]["histo_normalization"],
        std_target=config["data"]["std_target"],
    )

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        # collate_fn=MultimodalCTWSIDataset.collate_fn
    )

    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model with the weights
    model = madpe_resnet34(
        device,
        pretrained_rad_path=config["training"]["pretrained_rad_path"],
        pretrained_histo_path=config["training"]["pretrained_histo_path"],
        vol_depth=config["model"]["vol_depth"],
        vol_wh=config["model"]["vol_wh"],
        backbone_pretrained=config["model"]["backbone_pretrained"],
        backbone_grad=config["model"]["backbone_grad"],  # freeze everything
        backbone_unfreeze_layers=config["model"]["backbone_unfreeze_layers"],
        d_model=64,
        dim_hider=128,
        nhead=4,  # unfreeze some
    )
    model.to(device)

    # Load weights of model from checkpoint
    checkpoint_path = args.ckpt
    # if not Path(checkpoint_path).exists():
    #     print(f"No checkpoint found at {checkpoint_path}")
    #     return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")

    # Define output layers to extract features from
    output_layers = args.output_layers
    print(f"Extracting features from layers: {output_layers}")

    # Extract features
    features_dict, labels, modality_masks, patient_ids = extract_features(
        model, val_loader, device, output_layers
    )

    # Save features
    save_features(features_dict, labels, modality_masks, patient_ids, args.out_dir)

    print("Feature extraction completed successfully")
    return True


if __name__ == "__main__":
    main()
