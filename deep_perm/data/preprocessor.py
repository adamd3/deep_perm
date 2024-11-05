import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Data preprocessor for the cell painting dataset"""

    def __init__(self, threshold: float = 200):
        self.threshold = threshold
        self.scaler = StandardScaler()

    def handle_outliers(self, df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """Clip outliers in the dataset"""
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype.kind in "fc":
                mean, std = df_clean[col].mean(), df_clean[col].std()
                df_clean[col] = df_clean[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)
        return df_clean

    def prepare_data(
        self, predictors_df: pd.DataFrame, outcomes_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the data for training"""
        # Validation and merging
        if not isinstance(predictors_df, pd.DataFrame) or not isinstance(outcomes_df, pd.DataFrame):
            raise TypeError("Inputs must be pandas DataFrames")

        merged_df = pd.merge(predictors_df, outcomes_df, on="Smiles", how="inner")

        # Create binary outcome
        y = (merged_df["AVG_cells"] >= self.threshold).astype(int)
        smiles = merged_df["Smiles"]

        # Feature processing
        feature_cols = [col for col in merged_df.columns if col not in ["Smiles", "AVG_cells"]]
        X = merged_df[feature_cols]
        X = self.handle_outliers(X)
        X = self.scaler.fit_transform(X)

        return X.astype(np.float32), y.values.astype(np.float32), smiles.values
