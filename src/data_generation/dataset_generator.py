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
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None, api_key: Optional[str] = None):
        """
        Initialize the DatasetGenerator.
        
        Args:
            config_loader: Optional ConfigLoader instance. If None, creates a new one.
            api_key: Optional API key for the generator.
        """
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader
            
        self.cv_generator = CVGenerator(config_loader=self.config_loader, api_key=api_key)
        
        # Get output paths from config
        dataset_config = self.config_loader.get('dataset_generation', {})
        self.positive_output_path = dataset_config.get(
            'positive_output_path'
        )
        self.negative_output_path = dataset_config.get(
            'negative_output_path'
        )
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.positive_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.negative_output_path), exist_ok=True)
        
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
        Generate a positive and a hard negative CV for the given JD,
        and save them to the configured output files.
        
        Args:
            jd_text: The Job Description text.
            language: The language for the generated CV.
        """
        # 1. Generate Positive CV
        try:
            positive_cv = self.cv_generator.generate_positive_cv(jd_text, language)
            
            positive_entry = {
                "original_jd": jd_text,
                "positive": positive_cv.cv_text,
                "match": 1
            }
            # 2. Generate Hard Negative CV
            hard_negative_cv = self.cv_generator.generate_hard_negative_cv(
                jd_text, 
                positive_cv, 
                language
            )
            
            negative_entry = {
                "original_jd": jd_text,
                "hard_negative": hard_negative_cv.cv_text,
                "modification_type": hard_negative_cv.modification_type,
                "match": 0
            }
            
            # Write both entries only after successful generation of both
            self._append_to_jsonl(self.positive_output_path, positive_entry)
            self._append_to_jsonl(self.negative_output_path, negative_entry)
            
        except Exception as e:
            logger.error(f"Error generating dataset entry for JD: {jd_text[:50]}... Error: {e}")
            raise e
