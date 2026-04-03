from google import genai
from google.genai import types
import os
import random
from dotenv import load_dotenv
from typing import Optional, Tuple
from ..utils.config_loader import ConfigLoader
from .models import CandidateProfile
from .template_renderer import render_cv
from .personas import PERSONA_LIBRARY
from .templates import TEMPLATE_TRAIN, TEMPLATE_TEST
from ..utils.prompts import (
    GENERATE_POSITIVE_PROMPT,
    GENERATE_HARD_NEGATIVE_PROMPT_A,
    GENERATE_HARD_NEGATIVE_PROMPT_B,
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

    def generate_positive_cv(
        self,
        jd_text: str,
        language: str = "English",
        split: str = "train",
    ) -> Tuple[CandidateProfile, str]:
        """
        Generate a positive (matching) CV for a given Job Description.

        After the LLM produces a structured ``CandidateProfile``, a random
        persona and a template from the appropriate split are selected to
        render the profile into a realistic CV string.

        Args:
            jd_text: The Job Description text.
            language: The language for the generated CV.
            split: Which template pool to draw from — ``'train'`` (default)
                or ``'test'``.

        Returns:
            ``(profile, cv_text)`` — the raw ``CandidateProfile`` and the
            rendered CV string.  The profile is needed by the hard-negative
            generator; the text is written to the dataset.
        """
        prompt = GENERATE_POSITIVE_PROMPT.format(jd=jd_text, output_language=language)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=CandidateProfile
            )
        )

        if not response.parsed:
            raise ValueError("Failed to parse response from Gemini API")

        profile: CandidateProfile = response.parsed
        persona = random.choice(PERSONA_LIBRARY)
        template_pool = TEMPLATE_TEST if split == "test" else TEMPLATE_TRAIN
        template = random.choice(template_pool)
        cv_text = render_cv(profile, template_name=template, persona=persona)
        return profile, cv_text

    def generate_hard_negative_cv(
        self,
        jd_text: str,
        positive_cv: CandidateProfile,
        language: str = "English",
        poison_strategy: Optional[str] = None,
        split: str = "train",
    ) -> str:
        """
        Generate a hard negative (near-miss) CV based on a matching CV and JD.

        **Prompt variant** is selected randomly before the LLM call:
        - ``GENERATE_HARD_NEGATIVE_PROMPT_A`` (60 %): mutate the positive CV.
        - ``GENERATE_HARD_NEGATIVE_PROMPT_B`` (40 %): generate a fresh near-miss.

        A random dealbreaker strategy (A / B / C) is also chosen here unless
        ``poison_strategy`` is supplied by the caller.

        After the LLM returns a ``CandidateProfile``, a random persona and a
        template from the appropriate split are selected to render the profile
        into a realistic CV string.

        Args:
            jd_text: The Job Description text.
            positive_cv: The original matching CandidateProfile object.
            language: The language for the generated CV.
            poison_strategy: Optional override for the dealbreaker strategy
                (``'A'``, ``'B'``, or ``'C'``).  If ``None``, one is chosen
                at random.
            split: Which template pool to draw from — ``'train'`` (default)
                or ``'test'``.

        Returns:
            Rendered CV string.
        """
        from ..utils.prompts import DEALBREAKER_A, DEALBREAKER_B, DEALBREAKER_C

        DEALBREAKER_MAP = {
            "A": DEALBREAKER_A,
            "B": DEALBREAKER_B,
            "C": DEALBREAKER_C,
        }

        # --- Select prompt variant (60 % A / 40 % B) ---
        prompt_template = (
            GENERATE_HARD_NEGATIVE_PROMPT_A
            if random.random() < 0.60
            else GENERATE_HARD_NEGATIVE_PROMPT_B
        )

        # --- Select dealbreaker strategy ---
        if poison_strategy is None:
            poison_strategy = random.choice(list(DEALBREAKER_MAP.keys()))
        dealbreaker_text = DEALBREAKER_MAP[poison_strategy]

        prompt = prompt_template.format(
            jd=jd_text,
            positive_cv_text=positive_cv.model_dump_json(),
            output_language=language,
            poison_strategy=dealbreaker_text,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=CandidateProfile
            )
        )

        if not response.parsed:
            raise ValueError("Failed to parse response from Gemini API")

        profile: CandidateProfile = response.parsed
        persona = random.choice(PERSONA_LIBRARY)
        template_pool = TEMPLATE_TEST if split == "test" else TEMPLATE_TRAIN
        template = random.choice(template_pool)
        return render_cv(profile, template_name=template, persona=persona)

    def generate_profile(self, jd_text: str, language: str = "English") -> CandidateProfile:
        """
        Generate a structured candidate profile for a given Job Description.
        
        Args:
            jd_text: The Job Description text.
            language: The language for the generated profile.
            
        Returns:
            CandidateProfile object containing the structured candidate data.
        """
        prompt = GENERATE_PROFILE_PROMPT.format(jd=jd_text, output_language=language)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=CandidateProfile
            )
        )
        
        if not response.parsed:
             raise ValueError("Failed to parse response from Gemini API")

        return response.parsed