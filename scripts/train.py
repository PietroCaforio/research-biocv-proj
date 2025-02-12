import argparse
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb


sys.path.insert(0, "./")  # noqa: E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa: E402
from models.dpe.main_model import madpe_resnet34  # noqa: E402
from training.trainer import MultimodalTrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = json.load(f)
    return config


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(
        project=config["wandb"]["project_name"],
        name=args.experiment_name,
        config=config,
    )

    # Create datasets
    train_dataset = MultimodalCTWSIDataset(
        split="train",
        dataset_path=config["data"]["dataset_path"],
        patches_per_wsi=config["data"]["patches_per_wsi"],
        sampling_strategy=config["data"]["sampling_strategy"],
        missing_modality_prob=config["data"]["missing_modality_prob"],
        require_both_modalities=config["data"]["require_both_modalities"],
    )

    val_dataset = MultimodalCTWSIDataset(
        split="val",
        dataset_path=config["data"]["dataset_path"],
        patches_per_wsi=config["data"]["patches_per_wsi"],
        sampling_strategy=config["data"]["sampling_strategy"],
        missing_modality_prob=config["data"]["missing_modality_prob"],
        require_both_modalities=config["data"]["require_both_modalities"],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        # collate_fn=MultimodalCTWSIDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        # collate_fn=MultimodalCTWSIDataset.collate_fn
    )

    # Initialize model
    model = madpe_resnet34(
        check_point_path=config["model"]["checkpoint_path"],
        vol_depth=config["model"]["vol_depth"],
        vol_wh=config["model"]["vol_wh"],
    )
    model.to(device)

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        experiment_name=args.experiment_name,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
