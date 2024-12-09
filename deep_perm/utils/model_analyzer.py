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
        metrics_df = pd.DataFrame(self.metrics_per_run)

        # Calculate summary statistics
        summary_stats = {
            "metrics": {
                col: {
                    "mean": metrics_df[col].mean(),
                    "std": metrics_df[col].std(),
                    "min": metrics_df[col].min(),
                    "max": metrics_df[col].max(),
                    "median": metrics_df[col].median(),
                }
                for col in metrics_df.columns
            }
        }

        # Save metrics table
        metrics_summary = pd.DataFrame.from_dict(
            {col: summary_stats["metrics"][col] for col in metrics_df.columns}, orient="index"
        )
        metrics_summary.round(4).to_csv(self.base_output_dir / "metrics_summary.csv")

        # Save per-run metrics
        metrics_df.round(4).to_csv(self.base_output_dir / "metrics_per_run.csv")

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
        metrics = ["confidence", "aleatoric", "entropy", "variability"]
        stats = {}

        for metric in metrics:
            values = []
            for run_df in self.dataiq_metrics_per_run:
                if metric in run_df.columns:
                    values.append(run_df[metric].mean())

            if values:
                stats[metric] = {"mean": np.mean(values), "std": np.std(values)}

        return stats

    def _plot_metric_stability(self, output_dir: Path) -> None:
        """Plot stability of DataIQ metrics across runs for individual examples"""
        metrics = ["confidence", "aleatoric", "entropy", "variability"]

        # First, align examples across runs using SMILES as identifier
        aligned_metrics = {}
        for metric in metrics:
            # Create matrix where rows are examples and columns are runs
            metric_values = []
            for run_idx, run_df in enumerate(self.dataiq_metrics_per_run):
                values = run_df.set_index("smiles")[metric]
                values.name = f"run_{run_idx}"  # Name the series for clear column names
                metric_values.append(values)
            metric_matrix = pd.concat(metric_values, axis=1)
            aligned_metrics[metric] = metric_matrix

            # Save aligned metrics for this metric
            metric_matrix.to_csv(output_dir / f"{metric}_aligned.tsv", sep="\t")

            # Also save summary statistics (mean and std) for each example
            summary_df = pd.DataFrame(
                {
                    "mean": metric_matrix.mean(axis=1),
                    "std": metric_matrix.std(axis=1),
                    "min": metric_matrix.min(axis=1),
                    "max": metric_matrix.max(axis=1),
                    "median": metric_matrix.median(axis=1),
                }
            )
            summary_df.index.name = "smiles"
            summary_df.to_csv(output_dir / f"{metric}_summary.tsv", sep="\t")

        # 1. Plot standard deviation distribution for each metric
        plt.figure(figsize=(10, 6))
        std_data = []
        for metric in metrics:
            std_values = aligned_metrics[metric].std(axis=1)
            std_data.append({"Metric": metric, "Std Dev": std_values})
        std_df = pd.concat([pd.DataFrame(d) for d in std_data])

        # # check for duplicated columns in std_df
        # std_df = std_df.loc[:, ~std_df.columns.duplicated] if std_df.columns.duplicated().any() else std_df

        # print the names of the duplicated columns
        print(std_df.columns[std_df.columns.duplicated()])

        sns.violinplot(data=std_df, x="Metric", y="Std Dev")
        plt.title("Distribution of Per-Example Metric Stability Across Runs")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "metric_stability_distribution.png")
        plt.close()

        # 2. Create stability heatmap for most variable examples
        plt.figure(figsize=(12, 8))
        all_stds = pd.DataFrame({metric: aligned_metrics[metric].std(axis=1) for metric in metrics})

        # Find top 50 most variable examples across all metrics
        top_variable = all_stds.mean(axis=1).nlargest(50).index

        # Create heatmap data
        heatmap_data = pd.DataFrame()
        for metric in metrics:
            heatmap_data[f"{metric}_std"] = all_stds.loc[top_variable, metric]

        sns.heatmap(
            heatmap_data.T, cmap="YlOrRd", xticklabels=False, yticklabels=True, cbar_kws={"label": "Standard Deviation"}
        )
        plt.title("Stability Heatmap for Most Variable Examples")
        plt.tight_layout()
        plt.savefig(output_dir / "metric_stability_heatmap.png")
        plt.close()

        # 3. Plot metric consistency scatter
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        for idx, metric in enumerate(metrics):
            metric_values = aligned_metrics[metric]
            mean_values = metric_values.mean(axis=1)
            std_values = metric_values.std(axis=1)

            axes[idx].scatter(mean_values, std_values, alpha=0.3, s=20)
            axes[idx].set_xlabel(f"Mean {metric}")
            axes[idx].set_ylabel(f"Std Dev {metric}")
            axes[idx].set_title(f"{metric} Consistency")

            # Add trend line
            z = np.polyfit(mean_values, std_values, 1)
            p = np.poly1d(z)
            axes[idx].plot(mean_values, p(mean_values), "r--", alpha=0.8)

            # Calculate and display correlation
            corr = np.corrcoef(mean_values, std_values)[0, 1]
            axes[idx].text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=axes[idx].transAxes,
                bbox={"facecolor": "white", "alpha": 0.8},
            )

        plt.tight_layout()
        plt.savefig(output_dir / "metric_consistency_scatter.png")
        plt.close()

        # 4. Save stability statistics
        stability_stats = {
            metric: {
                "mean_std": float(all_stds[metric].mean()),
                "median_std": float(all_stds[metric].median()),
                "max_std": float(all_stds[metric].max()),
                "min_std": float(all_stds[metric].min()),
                "percent_stable": float((all_stds[metric] < all_stds[metric].median()).mean() * 100),
            }
            for metric in metrics
        }

        with open(output_dir / "stability_stats.json", "w") as f:
            json.dump(stability_stats, f, indent=4)

    def plot_results(self, output_dir: str) -> None:
        """Plot and save all analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_performance_metrics(output_dir)
        self._plot_group_distributions(output_dir)
        self._plot_uncertainty_correlations(output_dir)
        self._plot_uncertainty_by_group(output_dir)
        self._plot_metric_stability(output_dir)

        summary_stats = self.analyze_results()
        with open(output_dir / "summary_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=4, default=str)

    def _plot_performance_metrics(self, output_dir: Path) -> None:
        """Plot performance metrics across runs"""
        metrics_df = pd.DataFrame(self.metrics_per_run)
        key_metrics = ["accuracy", "auroc", "auprc", "f1"]

        plt.figure(figsize=(12, 6))

        # Create box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=metrics_df[key_metrics])
        plt.title("Distribution of Metrics Across Runs")
        plt.ylabel("Score")
        plt.xticks(rotation=45)

        # Create run trajectory plot
        plt.subplot(1, 2, 2)
        for metric in key_metrics:
            plt.plot(range(1, len(metrics_df) + 1), metrics_df[metric], marker="o", label=metric)
        plt.xlabel("Run")
        plt.ylabel("Score")
        plt.title("Metrics Trajectory Across Runs")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "metrics_across_runs.png")
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
        metrics = ["confidence", "aleatoric", "entropy", "variability"]

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
        metrics = ["confidence", "aleatoric", "entropy", "variability"]

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
