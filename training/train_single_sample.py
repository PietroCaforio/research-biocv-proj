import sys

sys.path.insert(0, "./")

import torch  # noqa E402
import torch.nn as nn  # noqa E402
import torch.optim as optim  # noqa E402
from torch.utils.data import DataLoader  # noqa E402
from data.unimodal3D import UnimodalCTDataset3D  # noqa E402
from data.unimodal_wsi3D import UnimodalWSIDataset3D  # noqa E402
from models.dpe.main_model import madpe_resnet34  # noqa E402
import numpy as np  # noqa E402
from pathlib import Path  # noqa E402
import matplotlib.pyplot as plt  # noqa E402


def train_single_sample(
    model,
    train_ct_loader,
    train_wsi_loader,
    val_ct_loader,
    val_wsi_loader,
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
    train_ct = next(iter(train_ct_loader))
    train_wsi = next(iter(train_wsi_loader))

    # Get single validation sample
    val_ct = next(iter(val_ct_loader))
    val_wsi = next(iter(val_wsi_loader))

    # Ensure we have matching patient IDs
    assert (
        train_ct["patient_id"] == train_wsi["patient_id"]
    ), "Training samples don't match"
    assert (
        val_ct["patient_id"] == val_wsi["patient_id"]
    ), "Validation samples don't match"

    # Convert to appropriate format and move to device
    train_ct_vol = (
        train_ct["volume"].float().unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)
    )
    train_wsi_vol = train_wsi["volume"].float().to(device)
    train_label = train_ct["label"].to(device)

    val_ct_vol = val_ct["volume"].float().unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)
    val_wsi_vol = val_wsi["volume"].float().to(device)
    val_label = val_ct["label"].to(device)

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print(f"Training on patient: {train_ct['patient_id'][0]}")
    print(f"Validating on patient: {val_ct['patient_id'][0]}")

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

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()

    return train_losses, val_losses


if __name__ == "__main__":
    # Initialize model and move to device
    madpe = madpe_resnet34()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    madpe.to(device)

    # Initialize datasets
    train_ct_dataset = UnimodalCTDataset3D(
        split="train", dataset_path="./data/processed/processed_CPTAC_PDA_71_3D"
    )
    train_wsi_dataset = UnimodalWSIDataset3D(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
    )

    val_ct_dataset = UnimodalCTDataset3D(
        split="val", dataset_path="./data/processed/processed_CPTAC_PDA_71_3D"
    )
    val_wsi_dataset = UnimodalWSIDataset3D(
        split="val",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
    )

    # Create dataloaders
    train_ct_loader = DataLoader(train_ct_dataset, batch_size=1, shuffle=True)
    train_wsi_loader = DataLoader(train_wsi_dataset, batch_size=1, shuffle=True)
    val_ct_loader = DataLoader(val_ct_dataset, batch_size=1, shuffle=True)
    val_wsi_loader = DataLoader(val_wsi_dataset, batch_size=1, shuffle=True)

    # Train the model
    train_losses, val_losses = train_single_sample(
        model=madpe,
        train_ct_loader=train_ct_loader,
        train_wsi_loader=train_wsi_loader,
        val_ct_loader=val_ct_loader,
        val_wsi_loader=val_wsi_loader,
        device=device,
        num_epochs=100,
        learning_rate=0.001,
        save_dir="training/checkpoints",
    )
