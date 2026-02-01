import pandas as pd
import os
import sys

# Add project root to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

def split_seed_data():
    """
    Splits the seed JD data into training and testing sets based on configuration.
    """
    logger.info("Loading configuration...")
    config_loader = ConfigLoader()
    split_config = config_loader.get('data_split', {})
    train_ratio = split_config.get('train_ratio', 0.9)
    
    input_path = os.path.join("scripts_output", "djinni_seed_jd.csv")
    train_output_path = os.path.join("scripts_output", "djinni_train_seed_jd.csv")
    test_output_path = os.path.join("scripts_output", "djinni_test_seed_jd.csv")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}")
        return

    logger.info(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    logger.info(f"Splitting data with ratio {train_ratio}...")
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Save
    logger.info(f"Saving train set to {train_output_path}...")
    train_df.to_csv(train_output_path, index=False)
    
    logger.info(f"Saving test set to {test_output_path}...")
    test_df.to_csv(test_output_path, index=False)
    
    logger.info("Data split complete.")

if __name__ == "__main__":
    split_seed_data()
