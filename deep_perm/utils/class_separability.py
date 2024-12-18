from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class ClassSeparabilityAnalyzer:
    """Class to analyze separability of features and classes in a dataset."""

    def __init__(self, features_df, y, output_dir=None):
        self.features_df = features_df
        self.feature_names = features_df.columns
        self.y = np.array(y)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # replace NA with median and scale features
        self.X = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median()).values
        self.X_scaled = StandardScaler().fit_transform(self.X)

    def _safe_division(self, a, b):
        """Safely divide two numbers, returning 0 if denominator is 0."""
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def _safe_histogram(self, data, bins=50):
        """Compute histogram safely handling edge cases."""
        # Remove any potential infinities
        data = data[~np.isinf(data)]

        if len(data) == 0:
            return np.zeros(bins), np.linspace(0, 1, bins + 1)

        if np.all(data == data[0]):
            # All values are identical
            edges = np.linspace(data[0] - 0.5, data[0] + 0.5, bins + 1)
            hist = np.zeros(bins)
            hist[bins // 2] = 1.0
            return hist, edges

        hist, edges = np.histogram(data, bins=bins, density=True)
        total = np.sum(hist * np.diff(edges))
        if total > 0:
            hist = hist / total
        return hist, edges

    def _calculate_overlap(self, dist1, dist2, bins=50):
        """Calculate overlap coefficient between two distributions."""
        # Compute histograms
        hist1, edges1 = self._safe_histogram(dist1, bins)
        hist2, edges2 = self._safe_histogram(dist2, bins)

        # Ensure both histograms use the same bins
        min_edge = min(edges1[0], edges2[0])
        max_edge = max(edges1[-1], edges2[-1])
        edges = np.linspace(min_edge, max_edge, bins + 1)

        hist1, _ = np.histogram(dist1, bins=edges, density=True)
        hist2, _ = np.histogram(dist2, bins=edges, density=True)

        # Normalize
        hist1 = self._safe_division(hist1, np.sum(hist1 * np.diff(edges)))
        hist2 = self._safe_division(hist2, np.sum(hist2 * np.diff(edges)))

        # Calculate overlap
        overlap = np.minimum(hist1, hist2).sum() * np.diff(edges)[0]
        return float(overlap)

    def analyze_feature_separability(self):
        """Analyze separability of individual features and plot distributions."""
        scores = []

        for i in range(self.X.shape[1]):
            feature_vals = self.X[:, i]
            class_0_vals = feature_vals[self.y == 0]
            class_1_vals = feature_vals[self.y == 1]

            fisher_score = float(self.fisher_score(i))
            mean_diff = abs(np.mean(class_1_vals) - np.mean(class_0_vals))
            pooled_std = np.sqrt((np.var(class_0_vals) + np.var(class_1_vals)) / 2)
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0
            overlap_coef = self._calculate_overlap(class_0_vals, class_1_vals)
            cohens_d = mean_diff / np.std(feature_vals) if np.std(feature_vals) > 0 else 0.0

            scores.append(
                {
                    "feature_name": self.feature_names[i],
                    "feature_idx": i,
                    "fisher_score": fisher_score,
                    "overlap_coefficient": overlap_coef,
                    "effect_size": effect_size,
                    "cohens_d": cohens_d,
                }
            )

        scores_df = pd.DataFrame(scores)

        # Create distribution plots
        plt.rcParams.update({"font.size": 14})
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Distribution of Feature Separability Metrics", fontsize=16, y=0.95)

        metrics = ["fisher_score", "effect_size", "overlap_coefficient", "cohens_d"]
        titles = ["Fisher Score", "Effect Size", "Overlap Coefficient", "Cohen's d"]

        for idx, (metric, title) in enumerate(zip(metrics, titles, strict=False)):
            ax = axes[idx // 2, idx % 2]
            sns.histplot(data=scores_df, x=metric, ax=ax, bins=20, alpha=0.6)

            # Add median line
            median = scores_df[metric].median()
            ax.axvline(median, color="red", linestyle="--", alpha=0.8)
            ax.text(
                0.98,
                0.95,
                f"Median: {median:.3f}",
                transform=ax.transAxes,
                horizontalalignment="right",
                bbox={"facecolor": "white", "alpha": 0.8},
            )

            ax.set_title(title)
            ax.set_xlabel(metric.replace("_", " ").title())

        plt.tight_layout()

        # Save the figure
        output_path = self.output_dir / "feature_metrics_distribution.png"
        plt.savefig(output_path)
        plt.close()

        return scores_df, fig

    def _calculate_mahalanobis(self, class_0_data, class_1_data):
        """Calculate Mahalanobis distance between class centroids."""
        # Calculate centroids
        centroid_0 = np.mean(class_0_data, axis=0)
        centroid_1 = np.mean(class_1_data, axis=0)

        # Pool covariance matrices
        cov_0 = np.cov(class_0_data, rowvar=False)
        cov_1 = np.cov(class_1_data, rowvar=False)
        n_0 = len(class_0_data)
        n_1 = len(class_1_data)
        pooled_cov = ((n_0 * cov_0) + (n_1 * cov_1)) / (n_0 + n_1)

        try:
            # Calculate Mahalanobis distance
            inv_covmat = np.linalg.inv(pooled_cov)
            diff = centroid_0 - centroid_1
            mahalanobis = np.sqrt(diff.dot(inv_covmat).dot(diff))
            return float(mahalanobis)
        except np.linalg.LinAlgError:
            return float("nan")

    def fisher_score(self, feature_idx):
        """Calculate Fisher Score for a feature."""
        X_subset = self.X[:, [feature_idx]]

        class_0_mask = self.y == 0
        class_1_mask = self.y == 1

        class_0_mean = np.mean(X_subset[class_0_mask])
        class_1_mean = np.mean(X_subset[class_1_mask])
        class_0_var = np.var(X_subset[class_0_mask])
        class_1_var = np.var(X_subset[class_1_mask])

        overall_mean = np.mean(X_subset)

        n0 = np.sum(class_0_mask)
        n1 = np.sum(class_1_mask)
        n_total = len(self.y)

        between_class_var = (
            n0 / n_total * (class_0_mean - overall_mean) ** 2 + n1 / n_total * (class_1_mean - overall_mean) ** 2
        )
        within_class_var = n0 / n_total * class_0_var + n1 / n_total * class_1_var

        within_class_var = max(within_class_var, 1e-10)
        return float(between_class_var / within_class_var)

    def analyze_dimensionality_reduction(self):
        """Analyze class separation using different dimensionality reduction techniques."""
        results = {}

        # PCA with full dimensions for silhouette
        pca = PCA()
        X_pca_full = pca.fit_transform(self.X_scaled)
        pca_silhouette = silhouette_score(X_pca_full, self.y)

        # PCA 2D for visualization
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(self.X_scaled)

        results["pca"] = {
            "coords": X_pca_2d,  # Keep 2D for visualization
            "explained_var": pca.explained_variance_ratio_,
            "silhouette_full": pca_silhouette,
            "cumulative_var": np.cumsum(pca.explained_variance_ratio_),
        }

        # Add minimum dimensions needed for X% variance explained
        thresholds = [0.8, 0.9, 0.95]
        dims_needed = {}
        for threshold in thresholds:
            dims = np.argmax(results["pca"]["cumulative_var"] >= threshold) + 1
            dims_needed[f"dims_for_{threshold:.0%}"] = int(dims)
        results["pca"]["dims_needed"] = dims_needed

        # Mahalanobis distance between classes
        class_0_data = self.X_scaled[self.y == 0]
        class_1_data = self.X_scaled[self.y == 1]
        mahalanobis_dist = self._calculate_mahalanobis(class_0_data, class_1_data)
        results["mahalanobis_distance"] = mahalanobis_dist

        # t-SNE with perplexity sensitivity
        perplexities = [5, 30, 50]
        tsne_results = {}
        for perp in perplexities:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
            X_tsne = tsne.fit_transform(self.X_scaled)
            tsne_results[f"perp_{perp}"] = {"coords": X_tsne, "silhouette": silhouette_score(X_tsne, self.y)}
        results["tsne"] = tsne_results

        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap = reducer.fit_transform(self.X_scaled)
        results["umap"] = {"coords": X_umap, "silhouette": silhouette_score(X_umap, self.y)}

        # LDA
        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(self.X_scaled, self.y).ravel()
        results["lda"] = {
            "coords": X_lda,  # Now 1D array
            "explained_var": lda.explained_variance_ratio_,
            "coef": lda.coef_,
            "intercept": lda.intercept_,
        }

        return results

    def plot_dimensionality_reduction(self, results):
        """Plot results from dimensionality reduction analysis."""
        plt.rcParams.update({"font.size": 14})
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        labels = ["Impermeant", "Permeant"]

        # PCA plot (top left)
        axes[0, 0].scatter(
            results["pca"]["coords"][self.y == 0, 0],
            results["pca"]["coords"][self.y == 0, 1],
            alpha=0.5,
            label=labels[0],
        )
        axes[0, 0].scatter(
            results["pca"]["coords"][self.y == 1, 0],
            results["pca"]["coords"][self.y == 1, 1],
            alpha=0.5,
            label=labels[1],
        )
        axes[0, 0].set_title(
            f"PCA\nSilhouette (full): {results['pca']['silhouette_full']:.2f}\n"
            f"Dims for 90% var: {results['pca']['dims_needed']['dims_for_90%']}",
        )
        axes[0, 0].legend()

        # Cumulative variance plot (top right)
        axes[0, 1].plot(np.arange(1, len(results["pca"]["cumulative_var"]) + 1), results["pca"]["cumulative_var"], "-o")
        axes[0, 1].set_xlabel("Number of Components")
        axes[0, 1].set_ylabel("Cumulative Explained Variance")
        axes[0, 1].set_title("PCA Cumulative Variance")
        axes[0, 1].grid(True)

        # t-SNE plot (bottom left)
        tsne_data = results["tsne"]["perp_30"]
        axes[1, 0].scatter(
            tsne_data["coords"][self.y == 0, 0],
            tsne_data["coords"][self.y == 0, 1],
            alpha=0.5,
            label=labels[0],
        )
        axes[1, 0].scatter(
            tsne_data["coords"][self.y == 1, 0],
            tsne_data["coords"][self.y == 1, 1],
            alpha=0.5,
            label=labels[1],
        )
        axes[1, 0].set_title(f"t-SNE\nSilhouette: {tsne_data['silhouette']:.2f}")
        axes[1, 0].legend()

        # LDA distribution plot (bottom right)
        lda_coords = results["lda"]["coords"].ravel()
        sns.kdeplot(data=lda_coords[self.y == 0], ax=axes[1, 1], label=labels[0], fill=True, alpha=0.5)
        sns.kdeplot(data=lda_coords[self.y == 1], ax=axes[1, 1], label=labels[1], fill=True, alpha=0.5)
        axes[1, 1].set_title("LDA Projection")
        axes[1, 1].set_xlabel("LDA Component")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def analyze_class_overlap(self):
        """Analyze overlap between classes using various metrics."""
        # Calculate feature-level separability
        self.analyze_feature_separability()

        # Calculate centroid distance
        class_0_centroid = np.mean(self.X_scaled[self.y == 0], axis=0)
        class_1_centroid = np.mean(self.X_scaled[self.y == 1], axis=0)
        centroid_distance = np.linalg.norm(class_1_centroid - class_0_centroid)

        # Calculate average within-class distance
        class_0_distances = np.mean(cdist(self.X_scaled[self.y == 0], [class_0_centroid]))
        class_1_distances = np.mean(cdist(self.X_scaled[self.y == 1], [class_1_centroid]))
        avg_within_class_distance = (class_0_distances + class_1_distances) / 2

        separation_ratio = centroid_distance / avg_within_class_distance if avg_within_class_distance > 0 else 0.0

        return {
            "centroid_distance": float(centroid_distance),
            "avg_within_class_distance": float(avg_within_class_distance),
            "separation_ratio": float(separation_ratio),
        }
