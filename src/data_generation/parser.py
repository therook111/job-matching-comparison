from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from ..utils.config_loader import ConfigLoader

# Load environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class DataParser:
    """
    A generic parser that calls the Gemini API to extract structured JSON output
    from a given text, using a provided Pydantic schema as the response schema.
    """

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the DataParser.

        Args:
            config_loader: Optional ConfigLoader instance. If None, creates a new one.
            api_key: Optional API key. If not provided, tries to load from environment.
        """
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader

        # Get configuration — falls back to the same key used by CVGenerator
        gen_config = self.config_loader.get("generation_model", {})
        self.model_name = gen_config.get("model_name", "gemini-2.0-flash-exp")

        # Resolve API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables or passed as argument"
            )

        # Initialize client
        self.client = genai.Client(api_key=self.api_key)

    def parse(self, text: str, schema: Type[T], prompt_template: Optional[str] = None) -> T:
        """
        Extract structured data from *text* and return it as an instance of *schema*.

        Args:
            text: The raw input text to parse (e.g. a CV, JD, or any freeform document).
            schema: A Pydantic BaseModel subclass that defines the expected output structure.
                    Gemini will be instructed to return JSON that conforms to this schema.
            prompt_template: Optional prompt template string.  If provided it must contain
                             a ``{text}`` placeholder which will be formatted with *text*.
                             If omitted, a sensible default prompt is used.

        Returns:
            An instance of *schema* populated with the parsed values.

        Raises:
            ValueError: If the Gemini API returns a response that cannot be parsed into
                        the requested schema.
        """
        if prompt_template is not None:
            prompt = prompt_template.format(text=text)
        else:
            prompt = (
                "Extract the relevant information from the following text and return it "
                "as structured JSON that matches the requested schema.\n\n"
                f"Text:\n{text}"
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        if not response.parsed:
            raise ValueError(
                f"Failed to parse Gemini API response into schema '{schema.__name__}'"
            )

        return response.parsed
