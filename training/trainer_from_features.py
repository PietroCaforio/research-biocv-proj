import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, "./")

from torch.utils.data import DataLoader  # noqa E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa E402
from models.dpe.main_model import madpe_resnet34  # noqa E402

from .trainer import (
    BaseTrainer,
    per_class_accuracy,
    precision_per_class,
    recall_per_class,
    f1_per_class,
)


class FromFeaturesTrainer(BaseTrainer):
    """Specific trainer implementation for multimodal CT-WSI learning on extracted features"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add default metric functions if not provided
        if not self.metric_functions:
            self.metric_functions = {
                "accuracy": lambda outputs, targets: {
                    "accuracy": (torch.max(outputs, 1)[1] == targets)
                    .float()
                    .mean()
                    .item()
                },
                "per_class_accuracy": per_class_accuracy,
                "precision": precision_per_class,
                "recall": recall_per_class,
                "f1_score": f1_per_class,
            }

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process a batch of multimodal data."""
        return {
            "ct_feat": batch["ct_feature"].float().to(self.device),
            "wsi_feat": batch["wsi_feature"].float().to(self.device),
            "labels": batch["label"].to(self.device),
            "modality_mask": batch["modality_mask"].to(self.device),
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = defaultdict(float)
        # metrics = {
        #    "train_loss": 0.0,
        #    "train_accuracy": 0.0,
        #    "G1_TrainAcc": 0.0,
        #    "G2_TrainAcc": 0.0,
        #    "G3_TrainAcc": 0.0,
        # }
        num_batches = len(self.train_loader)
        total_outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        total_labels = torch.tensor([], dtype=torch.long, device=self.device)
        for batch_idx, batch in enumerate(self.train_loader):
            # Process batch
            batch_data = self.process_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                batch_data["ct_feat"],
                batch_data["wsi_feat"],
                modality_flag=batch_data["modality_mask"],
                output_layers=["classification", "adapted_rad", "adapted_histo"],
            )

            # Compute loss
            loss = self.criterion(outputs["classification"], batch_data["labels"])
            loss = (
                loss
                + 0.1
                * (
                    1
                    - F.cosine_similarity(
                        outputs["adapted_rad"], outputs["adapted_histo"]
                    )
                ).mean()
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            metrics["train_loss"] += loss.item()

            total_outputs = torch.cat((total_outputs, outputs["classification"]), dim=0)
            total_labels = torch.cat((total_labels, batch_data["labels"]), dim=0)
            # accuracy_metrics = self.metric_functions["accuracy"](
            #     outputs, batch_data["labels"]
            # )
            # class_metrics = self.metric_functions["per_class_accuracy"](
            #     outputs, batch_data["labels"]
            # )
            #
            # metrics["train_accuracy"] += accuracy_metrics["accuracy"]
            # for key, value in class_metrics.items():
            # metrics[f"{key.replace('Acc', 'TrainAcc')}"] += value

            # Log batch progress
            if (batch_idx + 1) % self.config["training"]["log_interval"] == 0:
                self.logger.info(
                    f"Epoch [{self.current_epoch+1}/{self.config['training']['num_epochs']}], "
                    f"Batch [{batch_idx+1}/{num_batches}], "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute additional metrics
        with torch.no_grad():
            for k, v in self.metric_functions.items():
                mtrc = v(total_outputs, total_labels)
                for kk, vv in mtrc.items():
                    metrics["train_" + kk] = vv
        metrics = dict(metrics)
        # Compute averages
        # for key in metrics:
        #    metrics[key] /= num_batches
        metrics["train_loss"] /= num_batches
        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics = defaultdict(float)
        # metrics = {
        #    "val_loss": 0.0,
        #    "val_accuracy": 0.0,
        #    "G1_ValAcc": 0.0,
        #    "G2_ValAcc": 0.0,
        #    "G3_ValAcc": 0.0,
        # }
        num_batches = len(self.val_loader)
        total_outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        total_labels = torch.tensor([], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for batch in self.val_loader:
                # Process batch
                batch_data = self.process_batch(batch)

                # Forward pass
                outputs = self.model(
                    batch_data["ct_feat"],
                    batch_data["wsi_feat"],
                    modality_flag=batch_data["modality_mask"],
                )["classification"]

                # Compute loss
                loss = self.criterion(outputs, batch_data["labels"])

                # Update metrics
                metrics["val_loss"] += loss.item()

                total_outputs = torch.cat((total_outputs, outputs), dim=0)
                total_labels = torch.cat((total_labels, batch_data["labels"]), dim=0)

                # accuracy_metrics = self.metric_functions["accuracy"](
                #     outputs, batch_data["labels"]
                # )
                # class_metrics = self.metric_functions["per_class_accuracy"](
                #     outputs, batch_data["labels"]
                # )
        #
        # metrics["val_accuracy"] += accuracy_metrics["accuracy"]
        # for key, value in class_metrics.items():
        #     metrics[f"{key.replace('Acc', 'ValAcc')}"] += value
        # Compute additional metrics
        for k, v in self.metric_functions.items():
            mtrc = v(total_outputs, total_labels)
            for kk, vv in mtrc.items():
                metrics["val_" + kk] = vv
        metrics = dict(metrics)

        metrics["val_loss"] /= num_batches
        # Compute averages
        # for key in metrics:
        #  metrics[key] /= num_batches

        return metrics
