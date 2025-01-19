import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df=None):
    """
    Preprocess the Garis Kemiskinan data by handling missing values and scaling numerical features.
    Returns scaled features and target (nilai values).
    """
    with mlflow.start_run(run_name="preprocessing"):
        # Load data if not provided
        if df is None:
            df = pd.read_csv("data/raw/dataset.csv")
            logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Log initial NaN counts
        logger.info(f"Initial NaN counts:\n{df.isna().sum()}")
        
        # Create numerical features
        df['year_num'] = df['tahun'] - df['tahun'].min()
        df['periode_num'] = (df['periode'] == 'SEPTEMBER').astype(int)
        
        # Create dummy variables for categorical columns
        df = pd.get_dummies(df, columns=['jenis', 'daerah'], drop_first=True)
        
        # Separate features and target
        features = ['year_num', 'periode_num'] + [col for col in df.columns if col.startswith(('jenis_', 'daerah_'))]
        X = df[features].copy()
        y = df['nilai'].copy()
        
        logger.info(f"Features selected: {features}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Handle missing values
        X = X.fillna(X.mean())  # Fill missing values in features with mean
        y = y.fillna(y.mean())  # Fill missing values in target with mean
        
        # Verify no NaNs remain
        if X.isna().any().any():
            logger.error("NaN values remain in features after filling!")
            raise ValueError("Failed to handle all NaN values in features")
        if y.isna().any():
            logger.error("NaN values remain in target after filling!")
            raise ValueError("Failed to handle all NaN values in target")
            
        logger.info("Successfully handled missing values")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Log preprocessing params
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("features", features)
        
        # Create processed data directory
        os.makedirs("data/processed", exist_ok=True)
        
        # Save processed data
        X_df = pd.DataFrame(X_scaled, columns=X.columns)
        y_df = pd.DataFrame({'nilai': y})
        
        # Final verification
        logger.info(f"Final X shape: {X_df.shape}, Final y shape: {y_df.shape}")
        logger.info(f"Any NaNs in X: {X_df.isna().any().any()}")
        logger.info(f"Any NaNs in y: {y_df.isna().any().any()}")
        
        X_df.to_csv("data/processed/features.csv", index=False)
        y_df.to_csv("data/processed/target.csv", index=False)
        
        return X_scaled, y.values

if __name__ == "__main__":
    preprocess_data() 