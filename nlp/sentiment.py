"""Sentiment analysis module for the AI Assistant."""

from typing import Dict, Optional, Any
from loguru import logger
from models.factory import ModelFactory

class SentimentAnalyzer:
    """Handles sentiment analysis using LLMs."""
    
    def __init__(self, model_factory: ModelFactory):
        """Initialize the sentiment analyzer.
        
        Args:
            model_factory (ModelFactory): Factory for creating LLM instances
        """
        self.model_factory = model_factory
        logger.info("Initialized sentiment analyzer")
    
    async def analyze(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze sentiment of the input text.
        
        Args:
            text (str): Text to analyze
            model_config (Optional[Dict[str, Any]]): Configuration for the model
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        try:
            # Get model with the specified configuration
            model_name = model_config.get("model_name") if model_config else None
            model = self.model_factory.get_model(model_name)
            
            prompt = f"""Analyze the sentiment of the following text and classify it as positive, negative, or neutral.
            Also provide a confidence score between 0 and 1.
            
            Text: {text}
            
            Format your response as JSON with the following structure:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": 0.XX,
                "explanation": "brief explanation of the sentiment"
            }}
            """
            
            response = await model.generate(prompt)
            
            # Extract the sentiment from the response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                "text": text,
                "analysis": response_text,
                "model_used": model.model_id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise