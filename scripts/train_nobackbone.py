import argparse
import json
import shutil
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "./")  # noqa: E402
from data.multimodal_features import MultimodalCTWSIDataset  # noqa: E402
from models.dpe.main_model_nobackbone import madpe_nobackbone  # noqa: E402
from training.trainer_from_features import FromFeaturesTrainer  # noqa: E402

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
    
    set_global_seed(seed=SEED)
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "gpu" in config["training"].keys():
        torch.cuda.set_device(config["training"]["gpu"])
        gpu_id = config["training"]["gpu"]
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    
    if not config["data"]["binary"]:
        # Create datasets
        train_dataset = MultimodalCTWSIDataset(
            split=config["data"]["train_split"],
            ct_path=config["data"]["ct_path"],
            wsi_path=config["data"]["wsi_path"],
            dataset_path=config["data"]["dataset_path"],
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
            ct_path=config["data"]["ct_path"],
            wsi_path=config["data"]["wsi_path"],
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
    model = madpe_nobackbone(
        rad_input_dim = config["model"]["rad_input_dim"],
        histo_input_dim = config["model"]["histo_input_dim"],
        inter_dim = config["model"]["inter_dim"],
        token_dim = config["model"]["token_dim"],
        num_classes = config["model"]["num_classes"]
    )
    model.to(device)
    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],weight_decay=0.01)
    if config["training"]["scheduler"]["type"] == "reduce_lr_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["training"]["scheduler"]["mode"],
            factor=config["training"]["scheduler"]["factor"],
            patience=config["training"]["scheduler"]["patience"],
            min_lr=config["training"]["scheduler"]["min_lr"],
        )
    elif config["training"]["scheduler"]["type"] == "cosine_annealing_lr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["scheduler"]["T_max"],
            eta_min=config["training"]["scheduler"]["eta_min"],
            last_epoch=config["training"]["scheduler"]["last_epoch"],
        )
    elif config["training"]["scheduler"]["type"] == "cosine_annealing_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config["training"]["scheduler"]["T_0"],
            T_mult = config["training"]["scheduler"]["T_mult"],
            eta_min = config["training"]["scheduler"]["eta_min"],
            last_epoch = config["training"]["scheduler"]["last_epoch"],            
            )
    
    if not config["data"]["binary"]:
        # Initialize trainer
        trainer = FromFeaturesTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            experiment_name=args.experiment_name,
            scheduler=scheduler,
            early_stopping=config["training"]["early_stopping"],
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
