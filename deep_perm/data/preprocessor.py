import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Class to preprocess data for model training"""

    def __init__(self, threshold=200):
        self.threshold = threshold
        self.scaler = StandardScaler()

    def prepare_data(self, predictors_df, outcomes_df, target_col="AVG_cells"):
        """Prepare data, keeping only predictor columns and target variable"""
        if not isinstance(predictors_df, pd.DataFrame) or not isinstance(outcomes_df, pd.DataFrame):
            raise TypeError("Inputs must be pandas DataFrames")

        # Convert column names to lowercase
        predictors_df.columns = predictors_df.columns.str.lower()
        outcomes_df.columns = outcomes_df.columns.str.lower()

        if "smiles" in predictors_df.columns and "smiles" in outcomes_df.columns:
            merged_df = pd.merge(predictors_df, outcomes_df, on="smiles", how="inner")
            print("merging on smiles")
        elif "name" in predictors_df.columns and "name" in outcomes_df.columns:
            merged_df = pd.merge(predictors_df, outcomes_df, on="name", how="inner")
            print("merging on name")
        else:
            raise ValueError("Neither 'smiles' nor 'name' columns are shared between the input files.")

        smiles = merged_df["smiles"]
        target = merged_df[target_col]

        # replace target_col values with binary values
        # based on threshold
        outcomes_df[target_col] = (outcomes_df[target_col] >= self.threshold).astype(int)

        feature_cols = [col for col in predictors_df.columns if col != "smiles" and col != "name"]
        X = merged_df[feature_cols]

        # check for duplicated columns
        duplicated_cols = X.columns[X.columns.duplicated()]
        if len(duplicated_cols) > 0:
            print(f"Duplicate columns found: {duplicated_cols}")

            # drop duplicated columns
            X = X.loc[:, ~X.columns.duplicated()]

        print("\nData Summary:")
        print(f"Number of samples: {len(merged_df)}")
        print(f"Number of features: {len(feature_cols)}")
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

        return X_scaled.astype(np.float32), y.values.astype(np.float32), smiles.values, outcomes_df
