"""LM Studio model implementation."""
import os
import json
import base64
import requests
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class LMStudioModel:
    """Model implementation for LM Studio API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LM Studio model.
        
        Args:
            config (Dict[str, Any]): Configuration for the model
        """
        self.model_id = config.get("model_id", "")
        self.api_url = config.get("api_url", "http://localhost:1234/v1")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        
        logger.info(f"Initialized LM Studio model: {self.model_id}")
    
    async def generate(self, prompt: str, **kwargs) -> Any:
        """Generate a response using the LM Studio API.
        
        Args:
            prompt (str): The prompt to generate from
            **kwargs: Additional parameters
            
        Returns:
            Any: The generated response
        """
        try:
            # Extract parameters from kwargs
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            # Check if this is a vision request
            is_vision = "image_data" in kwargs or kwargs.get("is_vision", False)
            image_path = kwargs.get("image_path", None)
            
            # Prepare the API request
            if is_vision and image_path:
                # Vision request with image
                return await self.generate_with_image(prompt, image_path, **kwargs)
            else:
                # Text-only request
                payload = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in ["image_data", "image_path", "is_vision"]:
                        payload[key] = value
                
                # Make the API request
                response = requests.post(
                    urljoin(self.api_url, "/chat/completions"),
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                
                # Check for errors
                if response.status_code != 200:
                    error_msg = f"LM Studio API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                
                # Parse the response
                result = response.json()
                
                # Extract the generated text
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    content = message.get("content", "")
                    
                    # Create a response object with a text attribute
                    class Response:
                        def __init__(self, text):
                            self.text = text
                    
                    return Response(content)
                else:
                    return "Error: No response generated"
                
        except Exception as e:
            error_msg = f"Error generating with LM Studio: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    async def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> Any:
        """Generate a response using the LM Studio API with an image.
        
        Args:
            prompt (str): The prompt to generate from
            image_path (str): Path to the image file
            **kwargs: Additional parameters
            
        Returns:
            Any: The generated response
        """
        try:
            # Extract parameters from kwargs
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
            
            # Prepare the message with image content
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }
            
            # Prepare the API request
            payload = {
                "model": self.model_id,
                "messages": [message],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in ["image_data", "image_path", "is_vision"]:
                    payload[key] = value
            
            # Make the API request
            response = requests.post(
                urljoin(self.api_url, "/chat/completions"),
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"LM Studio API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Parse the response
            result = response.json()
            
            # Extract the generated text
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                
                # Create a response object with a text attribute
                class Response:
                    def __init__(self, text):
                        self.text = text
                
                return Response(content)
            else:
                return "Error: No response generated"
            
        except Exception as e:
            error_msg = f"Error generating with LM Studio and image: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"