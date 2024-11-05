import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Class to preprocess data for model training"""

    def __init__(self, threshold=200):
        self.threshold = threshold
        self.scaler = StandardScaler()

    def prepare_data(self, predictors_df, outcomes_df):
        """Prepare data, keeping only predictor columns and target variable"""
        if not isinstance(predictors_df, pd.DataFrame) or not isinstance(outcomes_df, pd.DataFrame):
            raise TypeError("Inputs must be pandas DataFrames")

        merged_df = pd.merge(predictors_df, outcomes_df, on="Smiles", how="inner")

        smiles = merged_df["Smiles"]
        target = merged_df["AVG_cells"]

        feature_cols = [col for col in predictors_df.columns if col != "Smiles"]
        X = merged_df[feature_cols]

        print("\nData Summary:")
        print(f"Number of samples: {len(merged_df)}")
        print(f"Number of features: {len(feature_cols)}")
        print("\nFeature names:", feature_cols)
        print("\nFeature types:")
        print(X.dtypes)

        # Handle missing values in features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create binary outcome based on threshold
        y = (target >= self.threshold).astype(int)

        print("\nFinal shapes:")
        print(f"X: {X_scaled.shape}")
        print(f"y: {y.shape}")
        print(f"Class distribution: {np.mean(y):.2%} positive")

        return X_scaled.astype(np.float32), y.values.astype(np.float32), smiles.values
