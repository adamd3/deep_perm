import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from configs.model_config import ModelConfig
from data.dataset import PermeabilityDataset, create_balanced_loader
from data.preprocessor import DataPreprocessor
from models.permeability_net import PermeabilityNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from trainers.trainer import PermeabilityTrainer
from utils.logger import setup_logger
from utils.model_analyzer import ModelAnalyzer
from utils.visualization import VisualizationManager


def validate_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """Validate the quality of the splits"""
    from scipy.stats import ks_2samp

    print("\nValidating splits...")

    # Check feature distributions using KS test
    n_features = X_train.shape[1]
    ks_scores_train_val = []
    ks_scores_train_test = []

    for i in range(n_features):
        _, p_val = ks_2samp(X_train[:, i], X_val[:, i])
        ks_scores_train_val.append(p_val)

        _, p_val = ks_2samp(X_train[:, i], X_test[:, i])
        ks_scores_train_test.append(p_val)

    print("\nFeature distribution tests:")
    print(f"Mean KS p-value (train-val): {np.mean(ks_scores_train_val):.3f}")
    print(f"Mean KS p-value (train-test): {np.mean(ks_scores_train_test):.3f}")

    # Flag potentially problematic features
    problem_features = np.where((np.array(ks_scores_train_val) < 0.05) | (np.array(ks_scores_train_test) < 0.05))[0]

    if len(problem_features) > 0:
        print(f"\nWarning: {len(problem_features)} features show significant distribution differences")
        print(f"Feature indices: {problem_features}")


def create_data_splits(X, y, smiles, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature matrix
        y: Target vector
        smiles: SMILES strings
        test_size: Proportion of dataset to include in test split
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility

    Returns
    -------
        Six arrays: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, smiles_temp, smiles_test, indices_temp, indices_test = train_test_split(
        X,
        y,
        smiles,
        np.arange(len(X)),  # Add indices to split
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: create validation set
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, smiles_train, smiles_val, indices_train, indices_val = train_test_split(
        X_temp,
        y_temp,
        smiles_temp,
        indices_temp,  # Add indices to split
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp,
    )

    # Print split sizes and proportions
    total_samples = len(X)
    print("\nData split summary:")
    print(f"Total samples: {total_samples}")
    print(f"Training: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

    # Analyze class distribution in splits
    print("\nClass distribution:")
    print(f"Training: {np.mean(y_train):.3f} positive")
    print(f"Validation: {np.mean(y_val):.3f} positive")
    print(f"Test: {np.mean(y_test):.3f} positive")

    validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        smiles_train,
        smiles_val,
        smiles_test,
        indices_train,
        indices_val,
        indices_test,
    )


def save_experiment_results(output_dir: str, metrics_per_epoch, groups, final_metrics, model, config, smiles_train):
    """Save all experiment results and metrics"""
    results_dir = Path(output_dir)

    # Save metrics
    metrics_file = results_dir / "final_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=4)

    # Save model
    model_file = results_dir / "final_model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": asdict(config)},
        model_file,  # Assuming config is a dataclass
    )

    # Save DataIQ results
    avg_confidence = np.mean(np.array(metrics_per_epoch["confidence"]), axis=0)
    avg_aleatoric = np.mean(np.array(metrics_per_epoch["aleatoric"]), axis=0)
    avg_entropy = np.mean(np.array(metrics_per_epoch["entropy"]), axis=0)
    avg_mi = np.mean(np.array(metrics_per_epoch["mi"]), axis=0)
    avg_variability = np.mean(np.array(metrics_per_epoch["variability"]), axis=0)

    dataiq_results = pd.DataFrame(
        {
            "smiles": smiles_train,
            "classification": groups,
            "confidence": avg_confidence,
            "aleatoric": avg_aleatoric,
            "entropy": avg_entropy,
            "mi": avg_mi,
            "variability": avg_variability,
        }
    )
    dataiq_results.to_csv(results_dir / "dataiq_results.csv", index=False)

    group_stats = {
        group: {"count": int(np.sum(groups == group)), "percentage": float(np.mean(groups == group) * 100)}
        for group in ["Easy", "Hard", "Ambiguous"]
    }
    with open(results_dir / "group_stats.json", "w") as f:
        json.dump(group_stats, f, indent=4)


# TODO: move this to the model_analyzer script
class FeatureImportanceAnalyzer:
    """Analyze feature importance based on ambiguity reduction"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.viz = VisualizationManager(output_dir)

    def _get_baseline_ambiguity(self, X, y, smiles, config, device):
        """Get baseline ambiguity percentage for given features"""
        # Create data splits
        splits = create_data_splits(X, y, smiles)
        X_train, X_val, X_test, y_train, y_val, y_test, smiles_train, smiles_val, smiles_test = splits

        # Create datasets and loaders
        train_dataset = PermeabilityDataset(X_train, y_train)
        val_dataset = PermeabilityDataset(X_val, y_val)
        test_dataset = PermeabilityDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        # Create temporary directory for this run
        temp_dir = self.output_dir / "temp_features"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        model = PermeabilityNet(ModelConfig(input_size=X.shape[1])).to(device)
        trainer = PermeabilityTrainer(model, config, device, temp_dir)
        dataiq, groups, _ = trainer.train(train_loader, val_loader, test_loader)

        # Calculate ambiguity percentage
        ambig_pct = (groups == "Ambiguous").mean() * 100
        return ambig_pct

    def analyze_feature_importance(self, X, y, smiles, feature_names, config, device):
        """Analyze how each feature contributes to reducing ambiguity"""
        results = []
        baseline_ambig_pct = self._get_baseline_ambiguity(X, y, smiles, config, device)

        # Analyze each feature
        for idx in range(X.shape[1]):
            # Create dataset with only this feature
            X_single = X[:, [idx]]
            ambig_pct = self._get_baseline_ambiguity(X_single, y, smiles, config, device)

            # Calculate ambiguity reduction
            reduction = baseline_ambig_pct - ambig_pct

            results.append(
                {
                    "feature_idx": idx,
                    "feature_name": feature_names[idx],
                    "baseline_ambiguity": baseline_ambig_pct,
                    "feature_ambiguity": ambig_pct,
                    "ambiguity_reduction": reduction,
                }
            )

        # Convert to DataFrame and sort by reduction
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("ambiguity_reduction", ascending=False)

        # Save results
        results_df.to_csv(self.output_dir / "feature_importance.csv", index=False)

        # Create visualizations - pass X to plotting method
        self._plot_feature_importance(results_df, X)

        return results_df

    def _plot_feature_importance(self, results_df, X):
        """Create visualizations for feature importance analysis"""
        # 1. Bar plot of ambiguity reduction
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x="feature_name", y="ambiguity_reduction", color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Feature")
        plt.ylabel("Ambiguity Reduction (%)")
        plt.title("Feature Importance Based on Ambiguity Reduction")
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance_bar.png")
        plt.close()

        # 2. Cumulative ambiguity reduction plot
        plt.figure(figsize=(10, 6))
        cumulative_reduction = np.cumsum(results_df["ambiguity_reduction"].values)
        plt.plot(range(1, len(cumulative_reduction) + 1), cumulative_reduction, marker="o")
        plt.xlabel("Number of Features")
        plt.ylabel("Cumulative Ambiguity Reduction (%)")
        plt.title("Cumulative Effect of Features on Ambiguity Reduction")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "cumulative_reduction.png")
        plt.close()

        # 3. Heatmap of top feature correlations
        top_features = results_df["feature_idx"].head(10).values
        correlation_matrix = np.corrcoef(X[:, top_features].T)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            xticklabels=results_df["feature_name"].head(10),
            yticklabels=results_df["feature_name"].head(10),
            annot=True,
            cmap="coolwarm",
            center=0,
        )
        plt.title("Correlation Between Top Ambiguity-Reducing Features")
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_features_correlation.png")
        plt.close()


warnings.filterwarnings(
    "ignore",
    message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.",
    category=FutureWarning,
    module="torch.nn.modules.module",
)


def main():
    """Main entry point for the permeability prediction experiment."""
    parser = argparse.ArgumentParser(description="Train permeability prediction model")
    parser.add_argument("--predictors", type=str, required=True, help="Path to predictors TSV file")
    parser.add_argument("--outcomes", type=str, required=True, help="Path to outcomes TSV file")
    parser.add_argument("--target-col", type=str, required=True, help="Target (outcome) variable column name")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument("--importance", action="store_true", help="Run feature importance analysis")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step", "onecycle"],
        help="Learning rate scheduler type",
    )
    # parser.add_argument(
    #     "--conf-upper", type=float, default=0.75, help="Upper confidence threshold for DataIQ classification"
    # )
    # parser.add_argument(
    #     "--conf-lower", type=float, default=0.25, help="Lower confidence threshold for DataIQ classification"
    # )
    # parser.add_argument(
    #     "--aleatoric-percentile", type=float, default=50, help="Percentile threshold for aleatoric uncertainty"
    # )
    parser.add_argument(
        "--dips-xthresh",
        type=float,
        default=0,
        help="X-threshold for DataIQ classification. Default (0) = choose automatically",
    )
    parser.add_argument("--dips-ythresh", type=float, default=0.2, help="Y-threshold for DataIQ classification")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of runs for analysis")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--balanced-sampling", action="store_true", help="Use balanced class sampling for training")

    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ModelAnalyzer(base_output_dir, args.n_runs)

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # logger = setup_logger(__name__, output_dir / "training.log")
    # logger.info(f"Starting experiment with args: {args}")
    # logger.info(f"Using device: {device}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(__name__, base_output_dir / "experiment.log")
    logger.info(f"Starting experiment with {args.n_runs} runs using device: {device}")

    for run in range(args.n_runs):
        # Set a different but deterministic seed for each run
        run_seed = args.base_seed + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        output_dir = base_output_dir / f"run_{run}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # logger = setup_logger(__name__, output_dir / "training.log")
        # logger.info(f"Starting run {run+1}/{args.n_runs}")

        run_logger = setup_logger(f"{__name__}.run_{run}", output_dir / "training.log")
        run_logger.info(f"Starting run {run+1}/{args.n_runs} with seed {run_seed}")

        try:
            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            preprocessor = DataPreprocessor()
            X, y, smiles, outcomes_df = preprocessor.prepare_data(
                pd.read_csv(args.predictors, sep="\t"), pd.read_csv(args.outcomes, sep="\t"), args.target_col
            )
            # Create config
            config = ModelConfig(
                input_size=X.shape[1],
                use_early_stopping=args.early_stopping,
                scheduler_type=args.scheduler,
                # conf_upper=args.conf_upper,
                # conf_lower=args.conf_lower,
                # aleatoric_percentile=args.aleatoric_percentile,
                dips_xthresh=args.dips_xthresh,
                dips_ythresh=args.dips_ythresh,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            logger.info(f"Created model config: {config}")

            logger.info("Creating data splits...")

            splits = create_data_splits(X, y, smiles, random_state=run_seed)
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                smiles_train,
                smiles_val,
                smiles_test,
                train_indices,
                val_indices,
                test_indices,
            ) = splits

            # Create datasets and loaders
            train_dataset = PermeabilityDataset(X_train, y_train)
            val_dataset = PermeabilityDataset(X_val, y_val)
            test_dataset = PermeabilityDataset(X_test, y_test)

            # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            if args.balanced_sampling:
                train_loader = create_balanced_loader(train_dataset, config.batch_size)
            else:
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

            val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

            logger.info("Initializing model and trainer...")
            model = PermeabilityNet(config).to(device)
            trainer = PermeabilityTrainer(
                model,
                config,
                device,
                output_dir,
                outcomes_df,
                train_indices,
                target_col=args.target_col,
                train_loader=train_loader,
            )

            logger.info("Starting training...")
            dataiq, groups, final_metrics, metrics_per_epoch = trainer.train(train_loader, val_loader, test_loader)

            # Log final metrics
            logger.info("Training completed. Final metrics:")
            for metric_name, value in final_metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")

            # # Generate feature distribution plots
            # logger.info("Generating feature distribution plots...")
            # viz = VisualizationManager(output_dir)
            # for feature_idx in range(X.shape[1]):
            #     viz.plot_feature_distributions(X_train, X_val, X_test, feature_idx=feature_idx)

            # Save all results
            logger.info("Saving experiment results...")
            save_experiment_results(output_dir, metrics_per_epoch, groups, final_metrics, model, config, smiles_train)

            analyzer.collect_run_results(run)

            # Print summary of DataIQ groups
            group_counts = {group: np.sum(groups == group) for group in ["Easy", "Hard", "Ambiguous"]}
            logger.info("\nDataIQ Group Summary:")
            for group, count in group_counts.items():
                percentage = (count / len(groups)) * 100
                logger.info(f"{group}: {count} samples ({percentage:.1f}%)")

            if args.importance:
                # Get feature names (assuming they're in your predictors file)
                feature_names = [
                    col
                    for col in pd.read_csv(args.predictors, sep="\t").columns
                    if col not in ["Smiles", "SMILES", "smiles", "Name", "name", "NAME"]
                ]

                # Analyze feature importance
                logger.info("Analyzing feature importance based on ambiguity reduction...")
                importance_analyzer = FeatureImportanceAnalyzer(output_dir)
                importance_results = importance_analyzer.analyze_feature_importance(
                    X, y, smiles, feature_names, config, device
                )

                # Log top features
                logger.info("\nTop 10 features for reducing ambiguity:")
                for _, row in importance_results.head(10).iterrows():
                    logger.info(
                        f"{row['feature_name']}: "
                        f"{row['ambiguity_reduction']:.2f}% reduction "
                        f"(from {row['baseline_ambiguity']:.1f}% to {row['feature_ambiguity']:.1f}%)"
                    )

        except Exception:
            logger.exception(f"An error occurred during run {run}:")
            continue

    # Analyze and visualize results
    logger.info("All runs completed. Analyzing results...")
    analysis_dir = base_output_dir / "analysis"
    analyzer.plot_results(analysis_dir)

    # Print summary statistics
    results = analyzer.analyze_results()
    logger.info("\nSummary of results across runs:")
    logger.info(
        f"Mean accuracy: {results['metrics']['accuracy']['mean']:.3f} ± {results['metrics']['accuracy']['std']:.3f}"
    )
    logger.info(f"Mean AUROC: {results['metrics']['auroc']['mean']:.3f} ± {results['metrics']['auroc']['std']:.3f}")
    logger.info(f"Mean AUPRC: {results['metrics']['auprc']['mean']:.3f} ± {results['metrics']['auprc']['std']:.3f}")
    logger.info(f"Mean F1: {results['metrics']['f1']['mean']:.3f} ± {results['metrics']['f1']['std']:.3f}")

    logger.info(f"Experiment completed. Results saved to: {base_output_dir}")


if __name__ == "__main__":
    main()
