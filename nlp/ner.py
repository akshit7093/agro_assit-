"""Named Entity Recognition module for the AI Assistant."""

from typing import Dict, List, Optional, Any
from loguru import logger
from models.factory import ModelFactory

class NamedEntityRecognizer:
    """Handles named entity recognition using LLMs."""
    
    def __init__(self, model_factory: ModelFactory):
        """Initialize the NER component.
        
        Args:
            model_factory (ModelFactory): Factory for creating LLM instances
        """
        self.model_factory = model_factory
        logger.info("Initialized Named Entity Recognizer")
    
    async def extract_entities(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract named entities from the input text.
        
        Args:
            text (str): Text to analyze
            model_config (Optional[Dict[str, Any]]): Configuration for the model
            
        Returns:
            Dict[str, Any]: Extracted entities and metadata
        """
        try:
            # Get model with the specified configuration
            model_name = model_config.get("model_name") if model_config else None
            model = self.model_factory.get_model(model_name)
            
            prompt = f"""Extract all named entities from the following text. 
            Categorize them as PERSON, ORGANIZATION, LOCATION, DATE, or OTHER.
            
            Text: {text}
            
            Format your response as JSON with the following structure:
            {{
                "entities": [
                    {{"text": "entity text", "type": "entity type", "start": start_position, "end": end_position}},
                    ...
                ]
            }}
            """
            
            response = await model.generate(prompt)
            
            # Extract the entities from the response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                "text": text,
                "entities": response_text,
                "model_used": model.model_id
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            raise