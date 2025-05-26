import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional

import torch
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, "./")

from torch.utils.data import DataLoader  # noqa E402
from data.multimodal3D import MultimodalCTWSIDataset  # noqa E402
from models.dpe.main_model import madpe_resnet34  # noqa E402
from .metrics import (
    per_class_accuracy,
    precision_per_class,
    recall_per_class,
    f1_per_class,
    per_class_accuracy_binary,
    precision_per_class_binary,
    recall_per_class_binary,
    f1_per_class_binary,
)


class BaseTrainer:
    """Flexible base trainer class that handles configurable training functionality."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
        experiment_name: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[Dict] = None,
        metric_functions: Optional[Dict[str, Callable]] = None,
        n_validations: int = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.experiment_name = experiment_name.replace("/", "_")
        self.scheduler = scheduler
        self.n_validations = n_validations
        # Early stopping configuration
        self.early_stopping_config = early_stopping or {}
        self.early_stopping_counter = 0
        self.best_monitor_metric = (
            float("inf")
            if self.early_stopping_config.get("mode") == "min"
            else float("-inf")
        )

        # Metric functions for tracking
        self.metric_functions = metric_functions or {}

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_metrics = {}
        wandb.login()
        wandb.init(
            project=config["wandb"]["project_name"],
            name=self.experiment_name,
            config=config,
            dir=config["training"]["checkpoint_dir"] + self.experiment_name,
        )

        # Setup logging and checkpoints
        self.setup_logging()

    def setup_logging(self):
        """Initialize logging and create checkpoint directory."""
        self.checkpoint_dir = Path(
            self.config["training"]["checkpoint_dir"] + self.experiment_name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    self.checkpoint_dir / f"{self.experiment_name}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def compute_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor, phase: str
    ) -> Dict[str, float]:
        """Compute all registered metrics."""
        metrics = {}
        for metric_name, metric_fn in self.metric_functions.items():
            try:
                metric_value = metric_fn(outputs, targets)
                metrics[f"{phase}_{metric_name}"] = metric_value
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric_name}: {str(e)}")
        return metrics

    def check_early_stopping(self, monitor_value: float) -> bool:
        """Check if training should stop early."""
        if not self.early_stopping_config:
            return False

        patience = self.early_stopping_config.get("patience", 0)
        mode = self.early_stopping_config.get("mode", "min")
        min_delta = self.early_stopping_config.get("min_delta", 0.0)

        improved = (
            mode == "min" and monitor_value < self.best_monitor_metric - min_delta
        ) or (mode == "max" and monitor_value > self.best_monitor_metric + min_delta)

        if improved:
            self.best_monitor_metric = monitor_value
            self.early_stopping_counter = 0
            return False

        self.early_stopping_counter += 1
        if self.early_stopping_counter >= patience:
            self.logger.info(
                f"Early stopping triggered after {patience} epochs without improvement"
            )
            return True
        return False

    def update_scheduler(self, monitor_value: Optional[float] = None):
        """Update learning rate scheduler."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()

    def save_checkpoint(self, is_best: bool = False, val_loss=True):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_loss": self.best_val_loss,
            "best_monitor_metric": self.best_monitor_metric,
            "config": self.config,
            "training_metrics": self.training_metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        if is_best and val_loss == False:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint to {best_path}")
        elif is_best:
            best_path = (
                self.checkpoint_dir / f"{self.experiment_name}_best_val_loss.pth"
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load training checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"

        if not Path(checkpoint_path).exists():
            self.logger.info(f"No checkpoint found at {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_monitor_metric = checkpoint.get(
            "best_monitor_metric", self.best_monitor_metric
        )
        self.training_metrics = checkpoint.get("training_metrics", {})

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return True

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Implement in child class
            raise NotImplementedError

        return epoch_metrics

    def validate(self):
        """Validate the model."""
        self.model.eval()
        epoch_metrics = {}

        with torch.no_grad():
            for batch in self.val_loader:
                # Implement in child class
                raise NotImplementedError

        return epoch_metrics

    def train(self):
        """Main training loop."""
        wandb.init(
            project=self.config["training"].get("wandb_project", "default_project"),
            name=self.experiment_name,
            config=self.config,
        )

        for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch()
            if train_metrics is None:
                self.logger.info("Detected NaN values, training stopped.")
                break
            # Validation phase
            if self.n_validations == None:
                val_metrics = self.validate()
            else:
                val_metrics_list = []
                for i in range(self.n_validations):
                    val_metrics_list.append(self.validate())

                # Calculate the average for each metric
                val_metrics = {}
                for metric_name in val_metrics_list[0].keys():
                    val_metrics[metric_name] = sum(
                        d[metric_name] for d in val_metrics_list
                    ) / len(val_metrics_list)

            # Combine metrics and log
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_metrics[epoch] = epoch_metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics["learning_rate"] = current_lr
            wandb.log(epoch_metrics)

            # Update learning rate scheduler
            monitor_metric = val_metrics.get(
                self.config["training"].get("monitor_metric", "val_loss")
            )
            val_loss = val_metrics.get("val_loss")
            self.update_scheduler(monitor_metric)
            # Save checkpoint
            is_best = (
                monitor_metric < self.best_monitor_metric
                if self.early_stopping_config.get("mode") == "min"
                else monitor_metric > self.best_monitor_metric
            )
            # Check early stopping
            if self.check_early_stopping(monitor_metric):
                break
            is_best_loss = val_loss < self.best_val_loss
            if is_best_loss:
                self.best_val_loss = val_loss
            self.save_checkpoint(is_best_loss)
            self.save_checkpoint(is_best, val_loss=False)

            self.logger.info(f"Best monitor metric: {self.best_monitor_metric}")
            self.logger.info(f"Best monitor metric: {self.best_monitor_metric}")

            # Log epoch summary
            self.logger.info(
                f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]} - Metrics: '
                + ", ".join([f"{k}: {v:.8f}" for k, v in epoch_metrics.items()])
            )
        wandb.finish()
        return {
            self.config["training"]["monitor_metric"]: self.best_monitor_metric,
            "val_loss": self.best_val_loss,
        }


class MultimodalTrainer(BaseTrainer):
    """Specific trainer implementation for multimodal CT-WSI learning."""

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
            "ct_vol": batch["ct_volume"]
            .float()
            .unsqueeze(1)
            .repeat(1, 3, 1, 1, 1)
            .to(self.device),
            "wsi_vol": batch["wsi_volume"].float().to(self.device),
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
                batch_data["ct_vol"],
                batch_data["wsi_vol"],
                modality_flag=batch_data["modality_mask"],
            )["classification"]

            # Compute loss
            loss = self.criterion(outputs, batch_data["labels"])

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            metrics["train_loss"] += loss.item()

            total_outputs = torch.cat((total_outputs, outputs), dim=0)
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
                    batch_data["ct_vol"],
                    batch_data["wsi_vol"],
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


class MultimodalBinaryTrainer(BaseTrainer):
    """Specific trainer implementation for multimodal CT-WSI learning."""

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
                "per_class_accuracy": per_class_accuracy_binary,
                "precision": precision_per_class_binary,
                "recall": recall_per_class_binary,
                "f1_score": f1_per_class_binary,
            }

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process a batch of multimodal data."""
        return {
            "ct_vol": batch["ct_volume"]
            .float()
            .unsqueeze(1)
            .repeat(1, 3, 1, 1, 1)
            .to(self.device),
            "wsi_vol": batch["wsi_volume"].float().to(self.device),
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
                batch_data["ct_vol"],
                batch_data["wsi_vol"],
                modality_flag=batch_data["modality_mask"],
            )["classification"]

            # Compute loss
            loss = self.criterion(outputs, batch_data["labels"])

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            metrics["train_loss"] += loss.item()

            total_outputs = torch.cat((total_outputs, outputs), dim=0)
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
                    batch_data["ct_vol"],
                    batch_data["wsi_vol"],
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
