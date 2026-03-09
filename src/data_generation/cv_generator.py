from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, Type
from ..utils.config_loader import ConfigLoader
from ..utils.data_schemas import GeneratedCV, PoisonedCV, GeneratedSummary
from ..utils.prompts import (
    GENERATE_POSITIVE_PROMPT, 
    GENERATE_HARD_NEGATIVE_PROMPT,
    GENERATE_PROFILE_PROMPT
)

# Load environment variables
load_dotenv()

class CVGenerator:
    """
    Generator for creating synthetic CV data (positive and hard negative examples)
    using Google's Gemini models.
    """
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None, api_key: Optional[str] = None):
        """
        Initialize the CVGenerator.
        
        Args:
            config_loader: Optional ConfigLoader instance. If None, creates a new one.
            api_key: Optional API key. If not provided, tries to load from environment.
        """
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader
            
        # Get configuration
        gen_config = self.config_loader.get('generation_model', {})
        self.model_name = gen_config.get('model_name', 'gemini-2.0-flash-exp')
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('GEMINI_API_KEY')
            
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or passed as argument")
            
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)

    def generate_positive_cv(self, jd_text: str, language: str = "English") -> GeneratedCV:
        """
        Generate a positive (matching) CV for a given Job Description.
        
        Args:
            jd_text: The Job Description text.
            language: The language for the generated CV.
            
        Returns:
            GeneratedCV object containing the candidate name and CV text.
        """
        prompt = GENERATE_POSITIVE_PROMPT.format(jd=jd_text, output_language=language)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=GeneratedCV
            )
        )
        
        if not response.parsed:
             raise ValueError("Failed to parse response from Gemini API")

        return response.parsed

    def generate_hard_negative_cv(self, 
                                 jd_text: str, 
                                 positive_cv: GeneratedCV, 
                                 language: str = "English",
                                 poison_strategy: str = "Strategy A") -> PoisonedCV:
        """
        Generate a hard negative (poisoned) CV based on a matching CV and JD.
        
        Args:
            jd_text: The Job Description text.
            positive_cv: The original matching GeneratedCV object.
            language: The language for the generated CV.
            
        Returns:
            PoisonedCV object containing the poisoned CV text and metadata.
        """
        prompt = GENERATE_HARD_NEGATIVE_PROMPT.format(
            jd=jd_text,
            positive_cv_text=positive_cv.cv_text,
            output_language=language,
            poison_strategy=poison_strategy
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=PoisonedCV
            )
        )
        
        if not response.parsed:
            raise ValueError("Failed to parse response from Gemini API")
            
        return response.parsed

    def generate_profile(self, jd_text: str, language: str = "English") -> GeneratedCV:
        """
        Generate a profile summary for a given Job Description.
        
        Args:
            jd_text: The Job Description text.
            language: The language for the generated profile.
            
        Returns:
            GeneratedSummary object containing the summarized text.
        """
        prompt = GENERATE_PROFILE_PROMPT.format(jd=jd_text, output_language=language)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=GeneratedSummary
            )
        )
        
        if not response.parsed:
             raise ValueError("Failed to parse response from Gemini API")

        return response.parsed