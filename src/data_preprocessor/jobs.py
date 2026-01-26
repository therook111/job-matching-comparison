"""
Data preprocessor for job descriptions.
"""

import pandas as pd
import numpy as np
import torch
from typing import Optional, List
from sentence_transformers import util
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


logger = get_logger(__name__)


class JobsPreprocessor:
    """Class to preprocess and clean job description data."""

    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize the JobsPreprocessor.

        Args:
            config_loader: ConfigLoader instance to access configuration.
        """
        self.config = config_loader
        self.preprocessor_config = self.config.get('preprocessor', {})
        self.model = self.config.load_sentence_transformer()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataframe to clean and deduplicate job descriptions.

        Args:
            df: Input dataframe containing job descriptions.

        Returns:
            Cleaned dataframe.
        """
        logger.info(f"Initial shape: {df.shape}")
        
        df = self._drop_exact_duplicates(df)
        logger.info(f"After dropping exact duplicates: {df.shape}")

        df = self._drop_empty_rows(df)
        logger.info(f"After dropping empty rows: {df.shape}")

        df = self._drop_short_descriptions(df)
        logger.info(f"After dropping short descriptions: {df.shape}")

        df = self._drop_duplicate_descriptions(df)
        logger.info(f"After dropping duplicate long descriptions: {df.shape}")
        
        df = self._drop_semantic_duplicates(df)
        logger.info(f"After semantic deduplication: {df.shape}")

        return df

    def _drop_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:

        return df.drop_duplicates(keep='first')

    def _drop_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.dropna(subset=['Long Description', 'Company Name'])
        df = df[df['Long Description'].str.strip() != '']
        df = df[df['Company Name'].str.strip() != '']
        return df

    def _drop_short_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:

        quantile = self.preprocessor_config.get('min_description_quantile', 0.05)
        
        lengths = df['Long Description'].str.len()
        
        threshold = lengths.quantile(quantile)

        logger.info(f"Dropping descriptions shorter than {threshold} characters (quantile {quantile})")
        
        return df[lengths >= threshold]

    def _drop_duplicate_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:

        return df.drop_duplicates(subset=['Long Description'], keep='first')

    def _drop_semantic_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:

        similarity_threshold = self.preprocessor_config.get('similarity_threshold', 0.95)
        
        descriptions = df['Long Description'].tolist()
        if not descriptions:
            return df
            
        logger.info("Encoding descriptions for semantic deduplication...")
        # Encode all descriptions
        embeddings = self.model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
        
        if len(embeddings) > 20000:
             logger.warning("Dataset large for full pair-wise check. This might be slow.")
        
        # Compute cosine similarity
        cos_scores = util.cos_sim(embeddings, embeddings)
        
        lower_tri = torch.tril(cos_scores, diagonal=-1)
        
        max_scores, _ = torch.max(lower_tri, dim=1)
        
        is_duplicate = max_scores > similarity_threshold
        
        keep_mask = ~is_duplicate.cpu().numpy()
        
        return df[keep_mask]

