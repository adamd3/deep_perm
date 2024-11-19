import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP  # Fixed UMAP import


class ClassSeparabilityAnalyzer:
    """Analyzes separability of classes in feature space using multiple methods."""

    def __init__(self, features_df, y):
        """Initialize analyzer with feature DataFrame and labels.

        Args:
            features_df: DataFrame with features
            y: Labels (n_samples,)
        """
        self.features_df = features_df
        # Remove any non-numeric columns (like SMILES)
        self.numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        self.X = features_df[self.numeric_cols].values
        self.y = y
        self.X_scaled = StandardScaler().fit_transform(self.X)

    def fisher_score(self, feature_idx=None):
        """Calculate Fisher Score for feature(s).

        Args:
            feature_idx: Optional index or list of indices for specific features
                       If None, calculates for all features
        """
        if feature_idx is None:
            feature_idx = range(self.X.shape[1])

        X_subset = self.X[:, feature_idx] if isinstance(feature_idx, list | range) else self.X[:, [feature_idx]]

        # Calculate class-wise means and variances
        classes = np.unique(self.y)
        class_means = np.array([X_subset[self.y == c].mean(axis=0) for c in classes])
        class_vars = np.array([X_subset[self.y == c].var(axis=0) for c in classes])

        # Calculate overall mean
        overall_mean = X_subset.mean(axis=0)

        # Calculate between-class and within-class variance
        between_class_var = np.sum(
            [
                (len(X_subset[self.y == c]) / len(X_subset)) * (m - overall_mean) ** 2
                for c, m in zip(classes, class_means, strict=False)
            ],
            axis=0,
        )
        within_class_var = np.sum(
            [(len(X_subset[self.y == c]) / len(X_subset)) * v for c, v in zip(classes, class_vars, strict=False)],
            axis=0,
        )

        # Handle zero division
        within_class_var = np.where(within_class_var == 0, 1e-10, within_class_var)

        return between_class_var / within_class_var

    def analyze_feature_separability(self):
        """Analyze separability of individual features."""
        scores = []
        feature_names = self.numeric_cols

        for i in range(self.X.shape[1]):
            fisher_score = float(self.fisher_score(i))  # Convert from numpy array to float
            class_means = [self.X[self.y == c, i].mean() for c in [0, 1]]
            class_stds = [self.X[self.y == c, i].std() for c in [0, 1]]
            overlap_coef = self._calculate_overlap(self.X[self.y == 0, i], self.X[self.y == 1, i])

            # Calculate effect size, handling potential division by zero
            pooled_std = np.sqrt((class_stds[0] ** 2 + class_stds[1] ** 2) / 2)
            effect_size = abs(class_means[1] - class_means[0]) / pooled_std if pooled_std > 0 else np.inf

            scores.append(
                {
                    "feature_name": feature_names[i],
                    "feature_idx": i,
                    "fisher_score": fisher_score,
                    "mean_diff": abs(class_means[1] - class_means[0]),
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
        reducer = UMAP(n_components=2, random_state=42)
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
