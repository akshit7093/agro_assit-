

from typing import Dict, Any, Optional
from loguru import logger
from models.openrouter import OpenRouterModel

class ModelFactory:
    """Factory class for creating and managing different model instances."""
    
    def __init__(self):
        """Initialize the model factory."""
        self.models = {}
        self.model_configs = {
            "gemini": {
                "model_name": "google/gemini-2.0-flash-lite-preview-02-05:free",
                "class": OpenRouterModel
            },
            "deepseek": {
                "model_name": "deepseek/deepseek-r1-distill-llama-70b:free",
                "class": OpenRouterModel
            },
            "qwen": {
                "model_name": "qwen/qwen2.5-vl-72b-instruct:free",
                "class": OpenRouterModel
            },
            "NVIDIA": {
                "model_name": "nvidia/llama-3.1-nemotron-70b-instruct:free",
                "class": OpenRouterModel
            }
        }
        logger.info("Initialized ModelFactory")
    
    def get_model(self, model_type: Optional[str] = None):
        """Get or create a model instance."""
        try:
            model_type = model_type or "gemini"
            
            if model_type in self.models:
                return self.models[model_type]
            
            if model_type in self.model_configs:
                config = self.model_configs[model_type]
                model = config["class"](model_id=config["model_name"])
                self.models[model_type] = model
                return model
            
            logger.warning(f"Model type {model_type} not found, defaulting to gemini")
            return self.get_model("gemini")
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            # Instead of returning None, raise the exception so it can be properly handled
            # This prevents NoneType being used in await expressions
            raise ValueError(f"Failed to initialize model {model_type}: {str(e)}")
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available model types and their configurations.
        
        Returns:
            Dict[str, str]: Dictionary of model types and their names
        """
        return {model_type: config["model_name"] 
                for model_type, config in self.model_configs.items()}
