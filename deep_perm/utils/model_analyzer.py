import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ModelAnalyzer:
    """Analyze model performance and DataIQ metrics across multiple runs"""

    def __init__(self, base_output_dir: str, n_runs: int = 10):
        self.base_output_dir = Path(base_output_dir)
        self.n_runs = n_runs
        self.metrics_per_run = []
        self.dataiq_metrics_per_run = []

    def collect_run_results(self, run_idx: int) -> None:
        """Collect results from a single run"""
        run_dir = self.base_output_dir / f"run_{run_idx}"

        # Load final metrics
        with open(run_dir / "final_metrics.json") as f:
            final_metrics = json.load(f)
        self.metrics_per_run.append(final_metrics)

        # Load DataIQ results
        dataiq_df = pd.read_csv(run_dir / "dataiq_results.csv")
        self.dataiq_metrics_per_run.append(dataiq_df)

    def analyze_results(self) -> dict:
        """Analyze results across all runs"""
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(self.metrics_per_run)

        # Calculate aggregate statistics
        summary_stats = {
            "metrics": {
                col: {
                    "mean": metrics_df[col].mean(),
                    "std": metrics_df[col].std(),
                    "min": metrics_df[col].min(),
                    "max": metrics_df[col].max(),
                }
                for col in metrics_df.columns
            },
            "group_distributions": self._analyze_group_distributions(),
            "uncertainty_stats": self._analyze_uncertainty_metrics(),
        }

        return summary_stats

    def _analyze_group_distributions(self) -> dict:
        """Analyze distribution of Easy/Ambiguous/Hard groups across runs"""
        group_stats = []

        for run_df in self.dataiq_metrics_per_run:
            counts = run_df["classification"].value_counts(normalize=True)
            group_stats.append(
                {"Easy": counts.get("Easy", 0), "Ambiguous": counts.get("Ambiguous", 0), "Hard": counts.get("Hard", 0)}
            )

        return pd.DataFrame(group_stats).agg(["mean", "std"]).to_dict()

    def _analyze_uncertainty_metrics(self) -> dict:
        """Analyze uncertainty metrics across runs"""
        metrics = ["confidence", "aleatoric", "entropy", "mi", "variability"]
        stats = {}

        for metric in metrics:
            values = []
            for run_df in self.dataiq_metrics_per_run:
                if metric in run_df.columns:
                    values.append(run_df[metric].mean())

            if values:
                stats[metric] = {"mean": np.mean(values), "std": np.std(values)}

        return stats

    def plot_results(self, output_dir: str) -> None:
        """Generate visualizations of results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_performance_metrics(output_dir)
        self._plot_group_distributions(output_dir)
        self._plot_uncertainty_correlations(output_dir)
        self._plot_uncertainty_by_group(output_dir)

        # Save summary statistics
        summary_stats = self.analyze_results()
        with open(output_dir / "summary_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=4, default=str)

    def _plot_performance_metrics(self, output_dir: Path) -> None:
        """Plot performance metrics across runs"""
        metrics_df = pd.DataFrame(self.metrics_per_run)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_df[["accuracy", "auroc", "auprc", "f1"]])
        plt.title("Model Performance Metrics Across Runs")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "performance_metrics.png")
        plt.close()

    def _plot_group_distributions(self, output_dir: Path) -> None:
        """Plot distribution of groups across runs"""
        group_data = []
        for run_df in self.dataiq_metrics_per_run:
            counts = run_df["classification"].value_counts(normalize=True)
            for group in ["Easy", "Ambiguous", "Hard"]:
                group_data.append({"Group": group, "Proportion": counts.get(group, 0)})

        group_df = pd.DataFrame(group_data)

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=group_df, x="Group", y="Proportion")
        plt.title("Distribution of Groups Across Runs")
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.savefig(output_dir / "group_distributions.png")
        plt.close()

    def _plot_uncertainty_correlations(self, output_dir: Path) -> None:
        """Plot correlation heatmap of uncertainty metrics"""
        # Combine all runs
        all_metrics = pd.concat(self.dataiq_metrics_per_run)
        metrics = ["confidence", "aleatoric", "entropy", "mi", "variability"]

        correlation_matrix = all_metrics[metrics].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Between Uncertainty Metrics")
        plt.tight_layout()
        plt.savefig(output_dir / "uncertainty_correlations.png")
        plt.close()

    def _plot_uncertainty_by_group(self, output_dir: Path) -> None:
        """Plot distribution of uncertainty metrics for each group"""
        all_metrics = pd.concat(self.dataiq_metrics_per_run)
        metrics = ["confidence", "aleatoric", "entropy", "mi", "variability"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i < len(axes):
                sns.boxplot(data=all_metrics, x="classification", y=metric, ax=axes[i])
                axes[i].set_title(f"{metric.capitalize()} by Group")
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

        if len(axes) > len(metrics):
            for i in range(len(metrics), len(axes)):
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(output_dir / "uncertainty_by_group.png")
        plt.close()
