"""Code generation and assistance module for the AI Assistant."""

from typing import Dict, Optional, Any
from loguru import logger
from models.factory import ModelFactory

class CodeAssistant:
    """Handles code generation and assistance using LLMs."""
    
    def __init__(self, model_factory: ModelFactory):
        """Initialize the code assistant.
        
        Args:
            model_factory (ModelFactory): Factory for creating LLM instances
        """
        self.model_factory = model_factory
        logger.info("Initialized Code Assistant")
    
    async def generate_code(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate code based on the input prompt.
        
        Args:
            text (str): Description of the code to generate
            model_config (Optional[Dict[str, Any]]): Configuration for the model
            
        Returns:
            Dict[str, Any]: Generated code and metadata
        """
        try:
            # Get model with the specified configuration
            model_name = model_config.get("model_name") if model_config else None
            model = self.model_factory.get_model(model_name)
            
            # Extract language if specified in the prompt
            language = None
            if "in python" in text.lower():
                language = "Python"
            elif "in javascript" in text.lower():
                language = "JavaScript"
            elif "in java" in text.lower():
                language = "Java"
            
            # Construct prompt
            prompt = f"Generate code for the following request:\n{text}"
            if language:
                prompt += f"\n\nPlease use {language} programming language."
            
            # Generate code
            response = await model.generate(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                "code": response_text,
                "language": language or "auto-detected",
                "prompt_length": len(text),
                "model_used": model.model_id
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise
    
    async def explain_code(self, code: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Explain the provided code snippet.
        
        Args:
            code (str): Code to explain
            model_config (Optional[Dict[str, Any]]): Configuration for the model
            
        Returns:
            Dict[str, Any]: Explanation and metadata
        """
        try:
            # Get model with the specified configuration
            model_name = model_config.get("model_name") if model_config else None
            model = self.model_factory.get_model(model_name)
            
            # Construct prompt
            prompt = f"Please explain the following code in detail:\n```\n{code}\n```"
            
            # Generate explanation
            response = await model.generate(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                "explanation": response_text,
                "code_length": len(code),
                "model_used": model.model_id
            }
            
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            raise
    
    def debug_code(self, code: str, error_message: Optional[str] = None) -> Dict[str, str]:
        """Debug the provided code snippet.
        
        Args:
            code (str): Code to debug
            error_message (Optional[str]): Error message if available
            
        Returns:
            Dict[str, str]: Debugging suggestions and fixed code
        """
        try:
            # Get LLM instance from factory
            model = self.model_factory.get_model()
            
            # Construct prompt
            prompt = f"Please debug the following code and provide a fixed version:\n```\n{code}\n```"
            if error_message:
                prompt += f"\n\nError message:\n{error_message}"
            
            # Generate debugging suggestions
            response = model.generate(prompt)
            
            return {
                "suggestions": response,
                "original_code": code
            }
            
        except Exception as e:
            logger.error(f"Error debugging code: {str(e)}")
            raise