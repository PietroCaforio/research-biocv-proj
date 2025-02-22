import argparse
import json
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

    # Create datasets
    train_dataset = MultimodalCTWSIDataset(
        split=config["data"]["train_split"],
        dataset_path=config["data"]["dataset_path"],
        patches_per_wsi=config["data"]["patches_per_wsi"],
        sampling_strategy=config["data"]["sampling_strategy"],
        missing_modality_prob=config["data"]["missing_modality_prob"],
        require_both_modalities=config["data"]["require_both_modalities"],
        pairing_mode=config["data"]["pairing_mode"],
        allow_repeats=config["data"]["allow_repeats"],
        pairs_per_patient=config["data"]["pairs_per_patient"],
        downsample=config["data"]["downsample"],
    )

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
        pretrained_rad_path=config["training"]["pretrained_rad_path"],
        pretrained_histo_path=config["training"]["pretrained_histo_path"],
        vol_depth=config["model"]["vol_depth"],
        vol_wh=config["model"]["vol_wh"],
        backbone_pretrained=True,
    )
    model.to(device)

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if config["training"]["scheduler"]["type"] == "reduce_lr_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["training"]["scheduler"]["mode"],
            factor=config["training"]["scheduler"]["factor"],
            patience=config["training"]["scheduler"]["patience"],
            min_lr=config["training"]["scheduler"]["min_lr"],
        )
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
        scheduler=scheduler,
    )

    shutil.copy(
        args.config, config["training"]["checkpoint_dir"] + args.experiment_name
    )
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
