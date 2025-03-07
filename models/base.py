"""Base interface for LLM model implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncIterator, Optional
from pydantic import BaseModel as PydanticBaseModel

class ModelResponse(PydanticBaseModel):
    """Standardized model response structure."""
    text: str
    tokens_used: int
    model_name: str
    metadata: Optional[Dict[str, Any]] = None

class BaseModel(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the model with configuration.
        
        Args:
            model_config (Dict[str, Any]): Model-specific configuration
        """
        self.config = model_config
        self.model_name = model_config.get('model_name', 'unknown')
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response for the given prompt.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse: Standardized response object
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream response tokens for the given prompt.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional model-specific parameters
            
        Yields:
            str: Response tokens as they are generated
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Calculate the number of tokens in the text.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of tokens
        """
        pass
    
    @abstractmethod
    def validate_response(self, response: str) -> bool:
        """Validate if the response meets quality criteria.
        
        Args:
            response (str): Model response to validate
            
        Returns:
            bool: True if response is valid, False otherwise
        """
        pass
    
    def prepare_prompt(self, template: str, **kwargs) -> str:
        """Prepare a prompt using a template and variables.
        
        Args:
            template (str): Prompt template
            **kwargs: Variables to fill in the template
            
        Returns:
            str: Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt template: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate the model configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['api_key', 'model_name']
        missing_fields = [field for field in required_fields if field not in self.config]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")
    
    async def generate_with_image(self, prompt: str, image_path: str) -> str:
        """Generate text based on a prompt and an image.
        
        Args:
            prompt (str): The text prompt
            image_path (str): Path to the image file
        
        Returns:
            str: The generated text
        """
        raise NotImplementedError("This model does not support image input")
