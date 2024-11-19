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


class ClassSeparabilityAnalyzer:
    """Class to analyze separability of classes in a dataset."""

    def __init__(self, features_df, y):
        self.features_df = features_df
        self.feature_names = features_df.columns
        self.X = features_df.values
        self.y = np.array(y)
        self.X_scaled = self.X

    def _calculate_overlap(self, dist1, dist2):
        """Calculate overlap coefficient between two distributions using min/max ranges."""
        # Remove any inf values
        dist1 = dist1[~np.isinf(dist1)]
        dist2 = dist2[~np.isinf(dist2)]

        # Check for empty arrays
        if len(dist1) == 0 or len(dist2) == 0:
            return 0.0

        # Calculate ranges
        min1, max1 = np.min(dist1), np.max(dist1)
        min2, max2 = np.min(dist2), np.max(dist2)

        # Calculate overlap range
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)

        # Calculate total range
        total_min = min(min1, min2)
        total_max = max(max1, max2)

        # If distributions are identical, return 1.0
        if total_min == total_max:
            return 1.0

        # Calculate overlap ratio
        overlap_range = max(0, overlap_max - overlap_min)
        total_range = total_max - total_min

        return overlap_range / total_range

    def analyze_feature_separability(self):
        """Analyze separability of individual features."""
        scores = []

        for i in range(self.X.shape[1]):
            feature_vals = self.X[:, i]
            class_0_vals = feature_vals[self.y == 0]
            class_1_vals = feature_vals[self.y == 1]

            # Calculate fisher score
            fisher_score = float(self.fisher_score(i))

            # Calculate mean difference
            mean_diff = abs(np.mean(class_1_vals) - np.mean(class_0_vals))

            # Calculate effect size
            pooled_std = np.sqrt((np.var(class_0_vals) + np.var(class_1_vals)) / 2)
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

            # Calculate overlap
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

    def fisher_score(self, feature_idx):
        """Calculate Fisher Score for a feature."""
        X_subset = self.X[:, [feature_idx]]

        class_0_mask = self.y == 0
        class_1_mask = self.y == 1

        class_0_mean = np.mean(X_subset[class_0_mask])
        class_1_mean = np.mean(X_subset[class_1_mask])
        class_0_var = np.var(X_subset[class_0_mask])
        class_1_var = np.var(X_subset[class_1_mask])

        # Calculate overall mean
        overall_mean = np.mean(X_subset)

        # Calculate between-class and within-class variance
        n0 = np.sum(class_0_mask)
        n1 = np.sum(class_1_mask)
        n_total = len(self.y)

        between_class_var = (
            n0 / n_total * (class_0_mean - overall_mean) ** 2 + n1 / n_total * (class_1_mean - overall_mean) ** 2
        )
        within_class_var = n0 / n_total * class_0_var + n1 / n_total * class_1_var

        # Handle zero division
        within_class_var = max(within_class_var, 1e-10)

        return between_class_var / within_class_var

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

        separation_ratio = centroid_distance / avg_within_class_distance if avg_within_class_distance > 0 else 0.0

        return {
            "centroid_distance": float(centroid_distance),
            "avg_within_class_distance": float(avg_within_class_distance),
            "separation_ratio": float(separation_ratio),
        }
