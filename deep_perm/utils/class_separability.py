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
    """Analyzes separability of classes in feature space using multiple methods."""

    def __init__(self, features_df, y):
        """
        Initialize analyzer with feature DataFrame and labels.

        Args:
            features_df: DataFrame with features (numeric only)
            y: Labels (n_samples,)
        """
        self.features_df = features_df
        self.feature_names = features_df.columns
        self.X = features_df.values  # Convert to numpy array
        self.y = np.array(y)  # Ensure y is numpy array
        self.X_scaled = StandardScaler().fit_transform(self.X)

    def fisher_score(self, feature_idx=None):
        """Calculate Fisher Score for feature(s)."""
        if feature_idx is None:
            feature_idx = range(self.X.shape[1])

        X_subset = self.X[:, feature_idx] if isinstance(feature_idx, list | range) else self.X[:, [feature_idx]]

        # Calculate class-wise means and variances
        class_0_mask = self.y == 0
        class_1_mask = self.y == 1

        class_0_mean = np.mean(X_subset[class_0_mask], axis=0)
        class_1_mean = np.mean(X_subset[class_1_mask], axis=0)
        class_0_var = np.var(X_subset[class_0_mask], axis=0)
        class_1_var = np.var(X_subset[class_1_mask], axis=0)

        # Calculate overall mean
        overall_mean = np.mean(X_subset, axis=0)

        # Calculate between-class and within-class variance
        n0 = np.sum(class_0_mask)
        n1 = np.sum(class_1_mask)
        n_total = len(self.y)

        between_class_var = (
            n0 / n_total * (class_0_mean - overall_mean) ** 2 + n1 / n_total * (class_1_mean - overall_mean) ** 2
        )
        within_class_var = n0 / n_total * class_0_var + n1 / n_total * class_1_var

        # Handle zero division
        within_class_var = np.where(within_class_var == 0, 1e-10, within_class_var)

        return between_class_var / within_class_var

    def analyze_feature_separability(self):
        """Analyze separability of individual features."""
        scores = []

        for i in range(self.X.shape[1]):
            # Get feature data
            feature_vals = self.X[:, i]
            class_0_vals = feature_vals[self.y == 0]
            class_1_vals = feature_vals[self.y == 1]

            # Calculate metrics
            fisher_score = float(self.fisher_score(i))
            mean_diff = abs(np.mean(class_1_vals) - np.mean(class_0_vals))

            # Calculate effect size
            pooled_std = np.sqrt((np.var(class_0_vals) + np.var(class_1_vals)) / 2)
            effect_size = mean_diff / pooled_std if pooled_std > 0 else np.inf

            # Calculate distribution overlap
            overlap_coef = self._calculate_overlap(class_0_vals, class_1_vals)

            scores.append(
                {
                    "feature_name": self.feature_names[i],
                    "feature_idx": i,
                    "fisher_score": fisher_score,
                    "mean_diff": mean_diff,
                    "overlap_coefficient": overlap_coef,
                    "effect_size": effect_size,
                }
            )

        return pd.DataFrame(scores)

    def _calculate_overlap(self, dist1, dist2, bins=50):
        """Calculate overlap coefficient between two distributions."""
        range_min = min(dist1.min(), dist2.min())
        range_max = max(dist1.max(), dist2.max())

        hist1, bins = np.histogram(dist1, bins=bins, range=(range_min, range_max), density=True)
        hist2, _ = np.histogram(dist2, bins=bins, range=(range_min, range_max), density=True)

        return np.minimum(hist1, hist2).sum() * (bins[1] - bins[0])

    def analyze_dimensionality_reduction(self):
        """Analyze class separation using different dimensionality reduction techniques."""
        results = {}

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        results["pca"] = {
            "coords": X_pca,
            "explained_var": pca.explained_variance_ratio_,
            "silhouette": silhouette_score(X_pca, self.y),
        }

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(self.X_scaled)
        results["tsne"] = {"coords": X_tsne, "silhouette": silhouette_score(X_tsne, self.y)}

        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap = reducer.fit_transform(self.X_scaled)
        results["umap"] = {"coords": X_umap, "silhouette": silhouette_score(X_umap, self.y)}

        # LDA
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(self.X_scaled, self.y)
        results["lda"] = {"coords": X_lda, "explained_var": lda.explained_variance_ratio_}

        return results

    def plot_dimensionality_reduction(self, results):
        """Plot results from dimensionality reduction analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # PCA plot
        axes[0, 0].scatter(
            results["pca"]["coords"][self.y == 0, 0],
            results["pca"]["coords"][self.y == 0, 1],
            alpha=0.5,
            label="Class 0",
        )
        axes[0, 0].scatter(
            results["pca"]["coords"][self.y == 1, 0],
            results["pca"]["coords"][self.y == 1, 1],
            alpha=0.5,
            label="Class 1",
        )
        axes[0, 0].set_title(
            f"PCA (Explained var: {results['pca']['explained_var'].sum():.2f})\n"
            f"Silhouette: {results['pca']['silhouette']:.2f}"
        )
        axes[0, 0].legend()

        # t-SNE plot
        axes[0, 1].scatter(
            results["tsne"]["coords"][self.y == 0, 0],
            results["tsne"]["coords"][self.y == 0, 1],
            alpha=0.5,
            label="Class 0",
        )
        axes[0, 1].scatter(
            results["tsne"]["coords"][self.y == 1, 0],
            results["tsne"]["coords"][self.y == 1, 1],
            alpha=0.5,
            label="Class 1",
        )
        axes[0, 1].set_title(f"t-SNE\nSilhouette: {results['tsne']['silhouette']:.2f}")
        axes[0, 1].legend()

        # UMAP plot
        axes[1, 0].scatter(
            results["umap"]["coords"][self.y == 0, 0],
            results["umap"]["coords"][self.y == 0, 1],
            alpha=0.5,
            label="Class 0",
        )
        axes[1, 0].scatter(
            results["umap"]["coords"][self.y == 1, 0],
            results["umap"]["coords"][self.y == 1, 1],
            alpha=0.5,
            label="Class 1",
        )
        axes[1, 0].set_title(f"UMAP\nSilhouette: {results['umap']['silhouette']:.2f}")
        axes[1, 0].legend()

        # LDA distribution plot
        sns.kdeplot(data=results["lda"]["coords"][self.y == 0].ravel(), ax=axes[1, 1], label="Class 0")
        sns.kdeplot(data=results["lda"]["coords"][self.y == 1].ravel(), ax=axes[1, 1], label="Class 1")
        axes[1, 1].set_title("LDA Projection")
        axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def calculate_bhattacharyya_distance(self):
        """Calculate Bhattacharyya distance between classes."""
        class_0 = self.X[self.y == 0]
        class_1 = self.X[self.y == 1]

        # Calculate means and covariances
        mean0 = np.mean(class_0, axis=0)
        mean1 = np.mean(class_1, axis=0)
        cov0 = np.cov(class_0.T)
        cov1 = np.cov(class_1.T)

        # Calculate average covariance
        cov_avg = (cov0 + cov1) / 2

        # Calculate first term (mean difference)
        mean_term = 1 / 8 * (mean1 - mean0).T @ np.linalg.inv(cov_avg) @ (mean1 - mean0)

        # Calculate second term (covariance difference)
        cov_term = 1 / 2 * np.log(np.linalg.det(cov_avg) / np.sqrt(np.linalg.det(cov0) * np.linalg.det(cov1)))

        return float(mean_term + cov_term)

    def analyze_class_overlap(self):
        """Analyze overlap between classes using various metrics."""
        # Calculate centroid distance
        class_0_centroid = np.mean(self.X_scaled[self.y == 0], axis=0)
        class_1_centroid = np.mean(self.X_scaled[self.y == 1], axis=0)
        centroid_distance = np.linalg.norm(class_1_centroid - class_0_centroid)

        # Calculate average within-class distance
        class_0_distances = np.mean(cdist(self.X_scaled[self.y == 0], [class_0_centroid]))
        class_1_distances = np.mean(cdist(self.X_scaled[self.y == 1], [class_1_centroid]))
        avg_within_class_distance = (class_0_distances + class_1_distances) / 2

        # Calculate Bhattacharyya distance
        bhattacharyya_dist = self.calculate_bhattacharyya_distance()

        return {
            "centroid_distance": centroid_distance,
            "avg_within_class_distance": avg_within_class_distance,
            "separation_ratio": centroid_distance / avg_within_class_distance,
            "bhattacharyya_distance": bhattacharyya_dist,
        }
