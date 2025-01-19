import pandas as pd
import mlflow
import os

def load_data():
    """
    Load Garis Kemiskinan (GK) dataset.
    Returns a pandas DataFrame with features and target.
    """
    # Start MLflow run
    with mlflow.start_run(run_name="data_loading"):
        # Load dataset
        df = pd.read_csv("dataset/gk.csv")
        
        # Melt the dataframe to get the expected format
        id_vars = ['provinsi']
        value_vars = [col for col in df.columns if col != 'provinsi']
        
        df_melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='category', value_name='nilai')
        
        # Extract components from category column
        df_melted[['jenis', 'daerah', 'tahun', 'periode']] = df_melted['category'].str.extract(r'nilai\.(\w+)\.(\w+)\.(\d+)\.(\w+)')
        
        # Convert tahun to numeric
        df_melted['tahun'] = pd.to_numeric(df_melted['tahun'])
        
        # Log dataset info
        mlflow.log_param("dataset_shape", df_melted.shape)
        mlflow.log_param("dataset_columns", list(df_melted.columns))
        mlflow.log_param("dataset_name", "garis_kemiskinan")
        
        # Create data directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        # Save raw data
        df_melted.to_csv("data/raw/dataset.csv", index=False)
        
        # Apply preprocessing
        df_processed = preprocess_data(df_melted)
        
        return df_processed

def preprocess_data(df):
    """
    Preprocess the input DataFrame by creating numerical features and encoding categorical variables.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess
    
    Returns:
        tuple: (X, y) where X is features DataFrame and y is target series
    """
    # Create numerical features
    df['year_num'] = df['tahun'] - df['tahun'].min()
    df['periode_num'] = (df['periode'] == 'SEPTEMBER').astype(int)
    
    # Create dummy variables for categorical columns
    df_encoded = pd.get_dummies(df, columns=['jenis', 'daerah'], drop_first=False)
    
    # Select features and target
    features = ['year_num', 'periode_num'] + [col for col in df_encoded.columns if col.startswith(('jenis_', 'daerah_'))]
    X = df_encoded[features].copy()  # Ensure a copy to avoid SettingWithCopyWarning
    y = df_encoded['nilai'].copy()
    
    return X, y

if __name__ == "__main__":
    load_data()