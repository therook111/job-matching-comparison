import json
import os
import threading
from typing import Optional, Dict, Any
from .cv_generator import CVGenerator
from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatasetGenerator:
    """
    Generator for creating dataset entries (positive and hard negative CVs)
    and saving them to JSONL files.
    """
    
    def __init__(
        self, 
        config_loader: Optional[ConfigLoader] = None, 
        api_key: Optional[str] = None, 
        split: str = "train"
    ):
        """
        Initialize the DatasetGenerator.
        
        Args:
            config_loader: Optional ConfigLoader instance. If None, creates a new one.
            api_key: Optional API key for the generator.
            split: Which data split to use ('train' or 'test'). 
                   Determines output paths from config.
        """
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader
            
        self.cv_generator = CVGenerator(config_loader=self.config_loader, api_key=api_key)
        
        # Get output paths from config based on split
        dataset_config = self.config_loader.get('dataset_generation', {})
        split_config = dataset_config.get(split, {})
        
        self.positive_output_path = split_config.get('positive_output_path')
        self.negative_output_path = split_config.get('negative_output_path')
        self.k_hard_negs = split_config.get('k_hard_negs', 1)
        
        if not self.positive_output_path or not self.negative_output_path:
            raise ValueError(
                f"Missing output paths for split '{split}' in dataset_generation config. "
                f"Expected 'dataset_generation.{split}.positive_output_path' and "
                f"'dataset_generation.{split}.negative_output_path'."
            )
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.positive_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.negative_output_path), exist_ok=True)
        
        self.split = split

        # Lock for file writing
        self.lock = threading.Lock()

    def _append_to_jsonl(self, file_path: str, data: Dict[str, Any]):
        """
        Append a dictionary as a JSON line to the specified file.
        
        Args:
            file_path: Path to the JSONL file.
            data: Dictionary to write.
        """
        with self.lock:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def generate_dataset_entry(self, jd_text: str, language: str = "English"):
        """
        Generate a positive CV and k hard negative CVs for the given JD,
        and save them to the configured output files.
        
        Args:
            jd_text: The Job Description text.
            language: The language for the generated CV.
        """
        try:
            positive_profile, positive_cv_text = self.cv_generator.generate_positive_cv(
                jd_text, language, split=self.split
            )

            positive_entry = {
                "original_jd": jd_text,
                "positive": positive_cv_text,
                "match": 1
            }

            negative_entries = []
            for i in range(self.k_hard_negs):
                hard_negative_text = self.cv_generator.generate_hard_negative_cv(
                    jd_text,
                    positive_profile,
                    language,
                    split=self.split,
                )

                negative_entries.append({
                    "original_jd": jd_text,
                    "hard_negative_index": i,
                    "hard_negative": hard_negative_text,
                    "match": 0
                })

            # Write all entries only after successful generation
            self._append_to_jsonl(self.positive_output_path, positive_entry)
            for neg_entry in negative_entries:
                self._append_to_jsonl(self.negative_output_path, neg_entry)
            
        except Exception as e:
            logger.error(f"Error generating dataset entry for JD: {jd_text[:50]}... Error: {e}")
            raise e
