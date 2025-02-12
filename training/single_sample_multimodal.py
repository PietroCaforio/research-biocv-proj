import sys

import wandb

sys.path.insert(0, "./")

import torch  # noqa E402
import torch.nn as nn  # noqa E402
import torch.optim as optim  # noqa E402
from torch.utils.data import DataLoader  # noqa E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa E402
from models.dpe.main_model import madpe_resnet34  # noqa E402
import numpy as np  # noqa E402
from pathlib import Path  # noqa E402
import matplotlib.pyplot as plt  # noqa E402


wandb.login()


def train_single_sample(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=0.001,
    save_dir="checkpoints",
):
    wandb.init(
        # Set the wandb project where this run will be logged
        project="multimodal_single_sample_training",
        name="SingleSampleWithMissingMode",
        # Track hyperparameters and run metadata
        config={"num_epochs": num_epochs, "lr": learning_rate},
    )
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Get single training sample
    train_sample = next(iter(train_loader))

    # Get single validation sample
    # val_sample = train_sample.copy()

    # Convert to appropriate format and move to device
    train_ct_vol = (
        train_sample["ct_volume"].float().unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)
    )
    train_wsi_vol = train_sample["wsi_volume"].float().to(device)
    train_label = train_sample["label"].to(device)

    val_ct_vol = (
        train_sample["ct_volume"].float().unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)
    )
    val_wsi_vol = train_sample["wsi_volume"].float().to(device)
    val_label = train_sample["label"].to(device)

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    modality_mask = train_sample["modality_mask"].to(device)

    print(f"Training on patient: {train_sample['patient_id'][0]}")
    print(f"Validating on patient: {train_sample['patient_id'][0]}")

    for epoch in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(train_ct_vol, train_wsi_vol, modality_flag=modality_mask)
        train_loss = criterion(outputs, train_label)

        train_loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, train_preds = torch.max(outputs, 1)
        train_correct = (train_preds == train_label).sum().item()
        train_accuracy = train_correct / train_label.size(0)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_ct_vol, val_wsi_vol, modality_flag=modality_mask)
            val_loss = criterion(val_outputs, val_label)

            # Calculate validation accuracy
            _, val_preds = torch.max(val_outputs, 1)
            val_correct = (val_preds == val_label).sum().item()
            val_accuracy = val_correct / val_label.size(0)

        # Save losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss.item(),
                    "val_loss": val_loss.item(),
                },
                save_path / "best_model.pth",
            )

        # Log metrics to wandb
        wandb.log(
            {
                "Train Loss": train_loss.item(),
                "Val Loss": val_loss.item(),
                "Train Accuracy": train_accuracy,
                "Val Accuracy": val_accuracy,
            }
        )

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(
                f"Train Loss: {train_loss.item():.4f}, Accuracy: {train_accuracy:.4f}, "
                f"Pred: {train_preds.item()}, True: {train_label.item()}"
            )
            print(
                f"Val Loss: {val_loss.item():.4f}, Accuracy: {val_accuracy:.4f}, "
                f"Pred: {val_preds.item()}, True: {val_label.item()}"
            )
            print("-" * 50)

    wandb.finish()
    return train_losses, val_losses


if __name__ == "__main__":
    # Initialize model and move to device
    madpe = madpe_resnet34(
        check_point_path="./models/pretrain_weights/r3d34_K_200ep.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    madpe.to(device)
    train_multimode_dataset = MultimodalCTWSIDataset(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D/",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
        missing_modality_prob=1.0,
    )

    # Create dataloaders
    # We use the same sample for train and validation
    train_multimode_loader = DataLoader(
        train_multimode_dataset, batch_size=1, shuffle=False
    )
    val_multimode_loader = DataLoader(
        train_multimode_dataset, batch_size=1, shuffle=False
    )

    # Train the model
    train_losses, val_losses = train_single_sample(
        model=madpe,
        train_loader=train_multimode_loader,
        val_loader=val_multimode_loader,
        device=device,
        num_epochs=300,
        learning_rate=0.001,
        save_dir="./models/ckpts/",
    )
