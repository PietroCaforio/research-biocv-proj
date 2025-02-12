import sys

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

sys.path.insert(0, "./")

from torch.utils.data import DataLoader  # noqa E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa: E402
from models.dpe.main_model import madpe_resnet34  # noqa: E402
from pathlib import Path  # noqa: E402

wandb.login()


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=0.001,
    save_dir="checkpoints",
):
    wandb.init(
        project="multimodal_training",
        name="FullDatasetTrainingBACKBONEGRAD",
        config={
            "num_epochs": num_epochs,
            "lr": learning_rate,
            "batch_size": train_loader.batch_size,
        },
    )

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        correct_per_class = [0, 0, 0]
        total_per_class = [0, 0, 0]

        for batch_idx, batch in enumerate(train_loader):
            ct_vol = (
                batch["ct_volume"].float().unsqueeze(1).repeat(1, 3, 1, 1, 1).to(device)
            )
            wsi_vol = batch["wsi_volume"].float().to(device)
            labels = batch["label"].to(device)
            modality_mask = batch["modality_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(ct_vol, wsi_vol, modality_flag=modality_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item()

            for i in range(3):
                correct_per_class[i] += ((preds == i) & (labels == i)).sum().item()
                total_per_class[i] += (labels == i).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        class_accuracy_train = [
            (100 * correct_per_class[i] / total_per_class[i])
            if total_per_class[i] > 0
            else 0
            for i in range(3)
        ]

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        correct_per_class = [0, 0, 0]
        total_per_class = [0, 0, 0]

        with torch.no_grad():
            for batch in val_loader:
                ct_vol = (
                    batch["ct_volume"]
                    .float()
                    .unsqueeze(1)
                    .repeat(1, 3, 1, 1, 1)
                    .to(device)
                )
                wsi_vol = batch["wsi_volume"].float().to(device)
                labels = batch["label"].to(device)
                modality_mask = batch["modality_mask"].to(device)

                outputs = model(ct_vol, wsi_vol, modality_flag=modality_mask)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()

                for i in range(3):
                    correct_per_class[i] += ((preds == i) & (labels == i)).sum().item()
                    total_per_class[i] += (labels == i).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        class_accuracy_val = [
            (100 * correct_per_class[i] / total_per_class[i])
            if total_per_class[i] > 0
            else 0
            for i in range(3)
        ]

        wandb.log(
            {
                "Train Loss": avg_train_loss,
                "Val Loss": avg_val_loss,
                "Train Accuracy": train_accuracy,
                "Val Accuracy": val_accuracy,
                "G1_TrainAcc": class_accuracy_train[0],
                "G2_TrainAcc": class_accuracy_train[1],
                "G3_TrainAcc": class_accuracy_train[2],
                "G1_ValAcc": class_accuracy_val[0],
                "G2_ValAcc": class_accuracy_val[1],
                "G3_ValAcc": class_accuracy_val[2],
                "Epoch": epoch + 1,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                save_path / "best_model.pth",
            )

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(
            f"Class Accuracies (Train) - \
                G1: {class_accuracy_train[0]:.2f}%, \
                G2: {class_accuracy_train[1]:.2f}%, \
                G3: {class_accuracy_train[2]:.2f}%"
        )
        print(
            f"Class Accuracies (Val) - \
                G1: {class_accuracy_val[0]:.2f}%, \
                G2: {class_accuracy_val[1]:.2f}%, \
                G3: {class_accuracy_val[2]:.2f}%"
        )
        print("-" * 50)

    wandb.finish()
    return model


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    madpe = madpe_resnet34(
        check_point_path="./models/pretrain_weights/r3d34_K_200ep.pth",
        vol_depth=66,
        vol_wh=224,
    )
    madpe.to(device)

    # Create datasets
    train_dataset = MultimodalCTWSIDataset(
        split="train",
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D/",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
        missing_modality_prob=0.0,
        require_both_modalities=True,
    )
    print("\n Train Dataset Statistics:")
    stats = train_dataset.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    val_dataset = MultimodalCTWSIDataset(
        split="val",  # Changed to use validation split
        dataset_path="./data/processed/processed_CPTAC_PDA_71_3D/",
        patches_per_wsi=66,
        sampling_strategy="consecutive",
        missing_modality_prob=0.0,
        require_both_modalities=True,
    )
    print("\n Validation Dataset Statistics:")
    stats = val_dataset.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Adjusted batch size
        shuffle=True,  # Enable shuffling
        num_workers=8,  # Enable parallel data loading
        pin_memory=True,  # Enable faster data transfer to GPU
    )

    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True
    )

    # Train the model
    trained_model = train_model(
        model=madpe,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,  # Adjusted number of epochs
        learning_rate=0.001,
        save_dir="./models/ckpts/",
    )
