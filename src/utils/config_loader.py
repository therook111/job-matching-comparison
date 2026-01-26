"""
Utility module for loading configuration and initializing models.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class ConfigLoader:
    """Utility class to load configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_path: Path to the config YAML file. If None, uses default path.
        """
        if config_path is None:
            # Default to the config.yaml in the resources directory
            self.config_path = Path(__file__).parent.parent / "resources" / "config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.
        
        Returns:
            Dictionary containing the configuration.
            
        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the config file is invalid YAML.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        return self.config
    
    def get_embedding_model_config(self) -> Dict[str, Any]:
        """
        Get the embedding model configuration.
        
        Returns:
            Dictionary containing embedding model settings.
        """
        return self.config.get('embedding_model', {})
    
    def load_sentence_transformer(self) -> SentenceTransformer:
        """
        Load and initialize a SentenceTransformer model based on the config.
        
        Returns:
            Initialized SentenceTransformer model.
            
        Raises:
            ValueError: If embedding_model configuration is missing or invalid.
        """
        embedding_config = self.get_embedding_model_config()
        
        if not embedding_config:
            raise ValueError("No 'embedding_model' configuration found in config file")
        
        model_name = embedding_config.get('name')
        if not model_name:
            raise ValueError("No 'name' specified in embedding_model configuration")
        
        device = embedding_config.get('device', 'cpu')
        
        # Initialize the SentenceTransformer model
        model = SentenceTransformer(model_name, device=device)
        
        return model
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: The configuration key (supports nested keys with dot notation).
            default: Default value if key is not found.
            
        Returns:
            The configuration value or default.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
