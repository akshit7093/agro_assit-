"""Text summarization module for the AI Assistant."""

from typing import Dict, Optional, Any
from loguru import logger
from models.factory import ModelFactory

import logging

logger = logging.getLogger(__name__)

class Summarizer:
    """Handles text summarization using LLMs."""
    
    def __init__(self, model_factory: ModelFactory):
        """Initialize the summarizer.
        
        Args:
            model_factory (ModelFactory): Factory for creating LLM instances
        """
        self.model_factory = model_factory
        logger.info("Initialized text summarizer")
    
    async def summarize(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a summary of the input text.
        
        Args:
            text (str): Text to summarize
            model_config (Optional[Dict[str, Any]]): Configuration for the model
            
        Returns:
            Dict[str, Any]: Summary and metadata
        """
        try:
            # Get model with the specified configuration
            model_name = model_config.get("model_name") if model_config else None
            model = self.model_factory.get_model(model_name)
            prompt = f"Please summarize the following text:\n\n{text}"
            
            response = await model.generate(prompt)
            
            return {
                "original_text": text,
                "summary": response.text,
                "summary_length": len(response.text),
                "model_used": response.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise