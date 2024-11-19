from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from data_iq.dataiq_class import DataIQ_Torch
from torch import optim
from tqdm import tqdm
from utils.dataiq_utils import classify_examples
from utils.logger import setup_logger
from utils.metrics import calculate_metrics
from utils.training_utils import update_metrics_per_epoch
from utils.visualization import VisualizationManager


class PermeabilityTrainer:
    """Trainer class for the permeability prediction model"""

    def __init__(
        self,
        model,
        config,
        device,
        output_dir,
        outcomes_df=None,
        train_indices=None,
        target_col=None,
        train_loader=None,
        use_weighted_loss=False,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)

        if outcomes_df is not None and train_indices is not None and target_col is not None:
            self.outcomes_df = outcomes_df.iloc[train_indices].reset_index(drop=True)
            labels = outcomes_df[target_col].values  # Use full dataset for weights

            # Calculate inverse frequency weights
            counts = np.bincount(labels)
            total = len(labels)
            weights = total / (len(counts) * counts)

            # Optional: Add scaling factor (e.g., 0.5 to dampen weights)
            scaling_factor = 0.5
            weights = weights * scaling_factor

            # With currentclass distribution (438 positive, 1818 negative), this
            # gives weights approximately:
            # Negative class: ~0.89 * 0.5 = 0.445
            # Positive class: ~3.68 * 0.5 = 1.84

            if use_weighted_loss:
                weights = torch.tensor(weights, dtype=torch.float32).to(device)
                self.criterion = nn.NLLLoss(weight=weights)
            else:
                self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.NLLLoss()
            self.outcomes_df = None

        # else:
        #     self.criterion = nn.NLLLoss()
        #     self.outcomes_df = None

        #     if outcomes_df is not None and train_indices is not None and target_col is not None:
        #         self.outcomes_df = outcomes_df.iloc[train_indices].reset_index(drop=True)
        #         train_labels = outcomes_df.iloc[train_indices][target_col].values
        #         # print(train_labels)
        #         neg_count = (train_labels == 0).sum()
        #         pos_count = (train_labels == 1).sum()
        #         pos_weight = np.sqrt(neg_count / pos_count)  # Square root to dampen the weight
        #         pos_weight = min(float(pos_weight), 10.0)  # Cap max
        #         weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(device)
        #         self.criterion = nn.NLLLoss(weight=weights)
        #     else:
        #         self.criterion = nn.NLLLoss()
        #         self.outcomes_df = None

        self.logger = setup_logger(__name__)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        if config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-6
            )
        elif config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)
        elif config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        elif config.scheduler_type == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.epochs,
                steps_per_epoch=len(train_loader),
            )

        self.metrics_per_epoch = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_auroc": [],
            "val_auprc": [],
            "val_f1": [],
            "confidence": [],
            "aleatoric": [],
            "correctness": [],
            "entropy": [],
            "mi": [],
            "variability": [],
        }

    def train(self, train_loader, val_loader, test_loader):
        """Train the model"""
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = self.output_dir / "best_model.pt"

        # Initialize DataIQ
        dataiq = DataIQ_Torch(X=train_loader.dataset.X.numpy(), y=train_loader.dataset.y.numpy(), sparse_labels=True)

        self.logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")

            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Update DataIQ
            dataiq.on_epoch_end(self.model, device=self.device)

            # Update metrics
            update_metrics_per_epoch(self.metrics_per_epoch, epoch, train_loss, train_acc, val_metrics, dataiq)

            # Early stopping logic...
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    best_model_path,
                )
            else:
                patience_counter += 1

            if self.config.use_early_stopping and patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            # Scheduler steps...
            if self.config.scheduler_type == "plateau":
                self.scheduler.step(val_metrics["loss"])
            elif self.config.scheduler_type != "onecycle":  # OneCycleLR scheduler steps after each batch
                self.scheduler.step()

        # Load best model for final evaluation
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Get final test metrics
        final_test_metrics = self.validate(test_loader)

        # Load visualization manager
        viz = VisualizationManager(self.output_dir)

        # Plot training metrics
        viz.plot_metrics(self.metrics_per_epoch, final_test_metrics)

        # DataIQ analysis
        aleatoric_vals = np.array(self.metrics_per_epoch["aleatoric"])
        avg_confidence = np.mean(np.array(self.metrics_per_epoch["confidence"]), axis=0)
        avg_aleatoric = np.mean(aleatoric_vals, axis=0)

        # dips_xthresh = 0.75 * (np.max(aleatoric_vals) - np.min(aleatoric_vals))
        # dips_ythresh = 0.2

        # if dips_xthresh is 0, choose 75% of the range of aleatoric uncertainty
        if self.config.dips_xthresh == 0:
            dips_xthresh = 0.75 * (np.max(aleatoric_vals) - np.min(aleatoric_vals))
        else:
            dips_xthresh = self.config.dips_xthresh

        self.logger.info(f"Using dips_xthresh: {dips_xthresh}")
        self.logger.info(f"Using dips_ythresh: {self.config.dips_ythresh}")

        groups = classify_examples(
            avg_confidence,
            avg_aleatoric,
            dips_xthresh=dips_xthresh,
            dips_ythresh=self.config.dips_ythresh,
        )

        # DataIQ visualizations
        viz.plot_dataiq_scatter(avg_confidence, avg_aleatoric, groups, self.outcomes_df)
        viz.plot_training_dynamics(self.metrics_per_epoch, groups)
        viz.plot_std_dev_relationships(avg_confidence, avg_aleatoric, self.outcomes_df)

        return dataiq, groups, final_test_metrics, self.metrics_per_epoch

    def train_epoch(self, train_loader):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0

        for X_batch, y_batch in tqdm(train_loader, desc="Training"):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).long()

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)

            loss = self.criterion(y_pred, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Step OneCycleLR scheduler after each batch
            if self.config.scheduler_type == "onecycle":
                self.scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total_acc += (predicted == y_batch).sum().item() / len(y_batch)

        return total_loss / len(train_loader), total_acc / len(train_loader)

    def validate(self, data_loader):
        """
        Validate the model on given data loader

        Args:
            data_loader: DataLoader for validation/test data
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).long()

                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                total_loss += loss.item()
                predictions.extend(y_pred.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())

        metrics = calculate_metrics(np.array(targets), np.array(predictions))
        metrics["loss"] = total_loss / len(data_loader)

        return metrics
