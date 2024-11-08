import os

import matplotlib.pyplot as plt
import numpy as np


class VisualizationManager:
    """Class to manage visualization of model training and data analysis"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_feature_distributions(self, X_train, X_val, X_test, feature_idx=0):
        """Plot distribution of features across splits with improved visualization"""
        plt.figure(figsize=(8, 6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        all_values = np.concatenate([X_train[:, feature_idx], X_val[:, feature_idx], X_test[:, feature_idx]])
        min_val = np.percentile(all_values, 1)
        max_val = np.percentile(all_values, 99)

        if np.std(all_values) < 1e-6:
            bins = np.linspace(np.min(all_values) - 0.1, np.max(all_values) + 0.1, 30)
        else:
            bins = np.linspace(min_val, max_val, 30)

        plt.hist(X_train[:, feature_idx], bins=bins, alpha=0.5, label="Train", color=colors[0], density=True)
        plt.hist(X_val[:, feature_idx], bins=bins, alpha=0.5, label="Val", color=colors[1], density=True)
        plt.hist(X_test[:, feature_idx], bins=bins, alpha=0.5, label="Test", color=colors[2], density=True)

        plt.axvline(np.mean(X_train[:, feature_idx]), color=colors[0], linestyle="--", alpha=0.8, label="Train mean")
        plt.axvline(np.mean(X_val[:, feature_idx]), color=colors[1], linestyle="--", alpha=0.8, label="Val mean")
        plt.axvline(np.mean(X_test[:, feature_idx]), color=colors[2], linestyle="--", alpha=0.8, label="Test mean")

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"feature_{feature_idx}_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_metrics(self, metrics_per_epoch, final_test_metrics):
        """Plot training metrics over epochs with final test performance"""
        plt.figure(figsize=(12, 8))
        epochs = metrics_per_epoch["epoch"]

        # Plot accuracy
        plt.subplot(2, 2, 1)
        self._plot_metric(
            epochs,
            metrics_per_epoch["train_accuracy"],
            metrics_per_epoch["val_accuracy"],
            final_test_metrics.get("accuracy"),
            "Accuracy",
        )

        # Plot AUROC
        plt.subplot(2, 2, 2)
        self._plot_metric(epochs, metrics_per_epoch["val_auroc"], None, final_test_metrics["auroc"], "AUROC")

        # Plot AUPRC
        plt.subplot(2, 2, 3)
        self._plot_metric(epochs, metrics_per_epoch["val_auprc"], None, final_test_metrics["auprc"], "AUPRC")

        # Plot F1
        plt.subplot(2, 2, 4)
        self._plot_metric(epochs, metrics_per_epoch["val_f1"], None, final_test_metrics["f1"], "F1 Score")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_metrics.png"))
        plt.close()

    def _plot_metric(self, epochs, train_values, val_values=None, test_value=None, metric_name="Metric"):
        plt.plot(epochs, train_values, "b-", label=f"Train {metric_name}")
        if val_values is not None:
            plt.plot(epochs, val_values, "r--", label=f"Val {metric_name}")
        if test_value is not None:
            plt.axhline(y=test_value, color="g", linestyle=":", label=f"Test {metric_name}: {test_value:.3f}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_dataiq_scatter(self, avg_confidence, avg_aleatoric, groups, outcomes_df=None):
        """Create scatter plot of aleatoric uncertainty vs confidence"""
        # Original scatter plot
        plt.figure(figsize=(7, 5))

        for group, color in zip(["Easy", "Hard", "Ambiguous"], ["green", "red", "blue"], strict=False):
            mask = groups == group
            plt.scatter(avg_aleatoric[mask], avg_confidence[mask], c=color, label=group, alpha=0.6, s=50)

        plt.xlabel("Aleatoric Uncertainty")
        plt.ylabel("Confidence")
        plt.title("DataIQ Classification")
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, "uncertainty_confidence.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # New scatter plot colored by cw_std_dev_norm if available
        if outcomes_df is not None and "cw_std_dev_norm" in outcomes_df.columns:
            # Verify we have the same number of samples
            if len(outcomes_df) != len(avg_confidence):
                print(
                    f"Warning: Mismatch in number of samples. outcomes_df: {len(outcomes_df)}, metrics: {len(avg_confidence)}"
                )
                return

            plt.figure(figsize=(7, 5))
            scatter = plt.scatter(
                avg_aleatoric, avg_confidence, c=outcomes_df["cw_std_dev_norm"], cmap="viridis", alpha=0.6, s=50
            )
            plt.colorbar(scatter, label="CW Std Dev Norm")
            plt.xlabel("Aleatoric Uncertainty")
            plt.ylabel("Confidence")
            plt.title("DataIQ Classification (Colored by CW Std Dev)")
            plt.tight_layout()

            plt.savefig(
                os.path.join(self.output_dir, "uncertainty_confidence_cw_std.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()

    def plot_training_dynamics(self, metrics_per_epoch, groups):
        """Analyze and plot how examples from different groups behave during training"""
        # Plot 1: Aleatoric uncertainty dynamics
        plt.figure(figsize=(10, 6))
        colors = {"Easy": "green", "Hard": "red", "Ambiguous": "blue"}

        for group in ["Easy", "Hard", "Ambiguous"]:
            group_indices = np.where(groups == group)[0]
            if len(group_indices) == 0:
                continue

            # Plot aleatoric uncertainty
            group_values = []
            for epoch in range(len(metrics_per_epoch["aleatoric"])):
                epoch_values = [metrics_per_epoch["aleatoric"][epoch][idx] for idx in group_indices]
                group_values.append(epoch_values)

            group_values = np.array(group_values)
            mean_values = np.mean(group_values, axis=1)
            std_values = np.std(group_values, axis=1)

            epochs = range(1, len(mean_values) + 1)
            plt.plot(epochs, mean_values, color=colors[group], label=f"{group}", linewidth=2)
            plt.fill_between(epochs, mean_values - std_values, mean_values + std_values, color=colors[group], alpha=0.2)

        plt.title("Aleatoric Uncertainty During Training by Group")
        plt.xlabel("Epoch")
        plt.ylabel("Aleatoric Uncertainty")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add summary statistics
        stats_text = []
        for group in ["Easy", "Hard", "Ambiguous"]:
            count = np.sum(groups == group)
            if count > 0:
                stats_text.append(f"{group}: {count} samples ({count/len(groups):.2%})")

        plt.text(
            0.02,
            0.98,
            "\n".join(stats_text),
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_dynamics_aleatoric.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 2: Confidence dynamics
        plt.figure(figsize=(10, 6))

        for group in ["Easy", "Hard", "Ambiguous"]:
            group_indices = np.where(groups == group)[0]
            if len(group_indices) == 0:
                continue

            # Plot confidence
            group_values = []
            for epoch in range(len(metrics_per_epoch["confidence"])):
                epoch_values = [metrics_per_epoch["confidence"][epoch][idx] for idx in group_indices]
                group_values.append(epoch_values)

            group_values = np.array(group_values)
            mean_values = np.mean(group_values, axis=1)
            std_values = np.std(group_values, axis=1)

            epochs = range(1, len(mean_values) + 1)
            plt.plot(epochs, mean_values, color=colors[group], label=f"{group}", linewidth=2)
            plt.fill_between(epochs, mean_values - std_values, mean_values + std_values, color=colors[group], alpha=0.2)

        plt.title("Confidence During Training by Group")
        plt.xlabel("Epoch")
        plt.ylabel("Confidence")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.text(
            0.02,
            0.98,
            "\n".join(stats_text),
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_dynamics_confidence.png"), dpi=300, bbox_inches="tight")
        plt.close()
