import sys

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


def train_single_sample(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=0.001,
    save_dir="checkpoints",
):
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
    print(val_wsi_vol.size())
    val_label = train_sample["label"].to(device)

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print(f"Training on patient: {train_sample['patient_id'][0]}")
    print(f"Validating on patient: {train_sample['patient_id'][0]}")

    for epoch in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(train_ct_vol, train_wsi_vol)
        train_loss = criterion(outputs, train_label)

        train_loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_ct_vol, val_wsi_vol)
            val_loss = criterion(val_outputs, val_label)

            # Get predictions
            _, train_preds = torch.max(outputs, 1)
            _, val_preds = torch.max(val_outputs, 1)

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

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(
                f"Train Loss: {train_loss.item():.4f}, Pred: {train_preds.item()}, True: {train_label.item()}"
            )
            print(
                f"Val Loss: {val_loss.item():.4f}, Pred: {val_preds.item()}, True: {val_label.item()}"
            )
            print("-" * 50)

    return train_losses, val_losses


if __name__ == "__main__":
    # Initialize model and move to device
    madpe = madpe_resnet34()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    madpe.to(device)
    train_multimode_dataset = MultimodalCTWSIDataset(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
        missing_modality_prob=0.0,
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
        num_epochs=100,
        learning_rate=0.001,
        save_dir="training/checkpoints",
    )
