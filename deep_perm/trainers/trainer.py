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


class PermeabilityTrainer:
    """Trainer class for the permeability prediction model"""

    def __init__(self, model, config, device, output_dir):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.logger = setup_logger(__name__)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)

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
        """
        Train the model

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = self.output_dir / "best_model.pt"

        # Initialize DataIQ
        dataiq = DataIQ_Torch(X=train_loader.dataset.X.numpy(), y=train_loader.dataset.y.numpy(), sparse_labels=True)

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Update DataIQ
            dataiq.on_epoch_end(self.model, device=self.device)

            # Update metrics
            update_metrics_per_epoch(self.metrics_per_epoch, epoch, train_loss, train_acc, val_metrics, dataiq)

            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUROC: {val_metrics['auroc']:.4f}"
            )

            # Early stopping
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

            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            self.scheduler.step()

        # Load best model for final evaluation
        checkpoint = torch.load(best_model_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Get final test metrics
        self.logger.info("Evaluating on test set...")
        final_test_metrics = self.validate(test_loader)
        self.logger.info("Test metrics:")
        for metric_name, value in final_test_metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")

        # DataIQ analysis
        aleatorics = np.array(dataiq.aleatoric)  # Shape should be [n_epochs, n_samples]
        confidences = np.array(dataiq.confidence)  # Shape should be [n_epochs, n_samples]

        avg_aleatoric = np.mean(aleatorics, axis=0)
        avg_confidence = np.mean(confidences, axis=0)
        groups = classify_examples(avg_confidence, avg_aleatoric)

        return dataiq, groups, final_test_metrics

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
