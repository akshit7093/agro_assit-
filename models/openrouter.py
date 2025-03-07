import os
import json
import asyncio
import base64  # Add this import for base64 encoding
from typing import Dict, List, Any, AsyncIterator, Optional
import aiohttp
from loguru import logger  # Fixed import syntax <button class="citation-flag" data-index="3">
from models.base import BaseModel, ModelResponse

class OpenRouterModel(BaseModel):
    """Implementation for OpenRouter API to access multiple LLMs."""
    
    # Add this at the class level, before __init__
    AVAILABLE_MODELS = {
        "gemini": "google/gemini-2.0-flash-lite-preview-02-05:free",
        "qwen": "qwen/qwen2.5-vl-72b-instruct:free",
        "NVIDIA": "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "deepseek": "deepseek/deepseek-r1-distill-llama-70b:free"
    }
    
    def __init__(self, model_id: str = "google/gemini-pro", **kwargs):
        """Initialize the OpenRouter model.
        
        Args:
            model_id (str): ID of the model to use
            **kwargs: Additional model parameters
        """
        self.model_id = model_id
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Default parameters
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 0.95)
        
        # Add missing config and template attributes
        self.config = {
            'site_url': kwargs.get('site_url', 'https://example.com'),
            'site_name': kwargs.get('site_name', 'AI Assistant'),
            'model_type': self._get_model_type(model_id),
            'api_key': self.api_key
        }
        
        self.template = {
            'system': kwargs.get('system_prompt', "You are a helpful AI assistant."),
            'user': kwargs.get('user_template', "{prompt}")
        }
        
        # Add API base URL
        self.api_base = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables")
    
    def _get_model_type(self, model_id: str) -> str:
        """Determine the model type from the model ID.
        
        Args:
            model_id (str): The model identifier
            
        Returns:
            str: The model type (gemini, qwen, llama, deepseek, or unknown)
        """
        if "gemini" in model_id.lower():
            return "gemini"
        elif "qwen" in model_id.lower():
            return "qwen"
        elif "llama" in model_id.lower():
            return "llama"
        elif "deepseek" in model_id.lower():
            return "deepseek"
        else:
            return "unknown"
            
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response for the given prompt."""
        formatted_prompt = self._format_prompt(prompt)
        payload = {
            "model": self.model_id,
            "messages": formatted_prompt,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
        }

        # Add optional parameters if provided
        if top_p := kwargs.get('top_p'):
            payload['top_p'] = top_p
        if frequency_penalty := kwargs.get('frequency_penalty'):
            payload['frequency_penalty'] = frequency_penalty
        if presence_penalty := kwargs.get('presence_penalty'):
            payload['presence_penalty'] = presence_penalty

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.get('site_url', ''),
            "X-Title": self.config.get('site_name', '')
        }

        # Add proper async context management
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                        raise Exception(f"API error: {response.status} - {error_text}")
                    
                    result = await response.json()

            # Validate and extract response
            if not result:
                raise ValueError("Empty response from API")
            
            if 'choices' not in result or not result['choices']:
                logger.error(f"Unexpected API response format: {json.dumps(result)}")
                raise ValueError("No choices in API response")
                
            try:
                response_text = result['choices'][0]['message']['content']
                tokens_used = result.get('usage', {}).get('total_tokens', 0)
                
                return ModelResponse(
                    text=response_text,
                    tokens_used=tokens_used,
                    model_name=self.model_id,
                    metadata={
                        "finish_reason": result['choices'][0].get('finish_reason'),
                        "raw_response": result
                    }
                )
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse API response: {str(e)}\nResponse: {json.dumps(result)}")
                raise ValueError(f"Invalid API response format: {str(e)}")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream response tokens for the given prompt."""
        formatted_prompt = self._format_prompt(prompt)
        payload = {
            "model": self.model_id,
            "messages": formatted_prompt,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": True
        }

        # Add optional parameters if provided
        if top_p := kwargs.get('top_p'):
            payload['top_p'] = top_p
        if frequency_penalty := kwargs.get('frequency_penalty'):
            payload['frequency_penalty'] = frequency_penalty
        if presence_penalty := kwargs.get('presence_penalty'):
            payload['presence_penalty'] = presence_penalty

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.get('site_url', ''),
            "X-Title": self.config.get('site_name', '')
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        line = line[6:]
                        if line == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            if content := data['choices'][0].get('delta', {}).get('content'):
                                yield content
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming response: {line}")
                            continue

    def get_token_count(self, text: str) -> int:
        """Calculate the number of tokens in the text (simplified)."""
        return len(text) // 4  # Simple estimation (~4 characters per token)

    def validate_response(self, response: str) -> bool:
        """Validate if the response meets quality criteria."""
        if not response or len(response.strip()) < 10:
            return False
        return True

    def _format_prompt(self, prompt: str, image_url: str = None) -> List[Dict[str, Any]]:
        """Format the prompt according to the model's requirements."""
        model_type = self.config.get('model_type', 'gemini')
        
        messages = [
            {"role": "system", "content": self.template.get("system", "You are a helpful AI assistant.")}
        ]
    
        user_template = self.template.get("user", "{prompt}")
        user_message = user_template.format(prompt=prompt)
        
        if image_url and model_type in ["gemini", "qwen"]:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_message})
        
        return messages
    async def generate_with_image(self, prompt: str, image_path: str) -> ModelResponse:
        """Generate text based on a prompt and an image.
        
        Args:
            prompt (str): The text prompt
            image_path (str): Path to the image file
            
        Returns:
            ModelResponse: The model's response
        """
        try:
            # Ensure the model supports vision
            if self.config.get("model_type") not in ["gemini", "qwen"]:
                raise ValueError(f"Model {self.config.get('model_type')} does not support vision")
            
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare the message with text and image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
            
            # Log the request structure (without te actual base64 data)
            debug_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,[BASE64_DATA]"}}
                    ]
                }
            ]
            logger.debug(f"Sending multimodal request: {json.dumps(debug_messages)}")
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.config['api_key']}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.AVAILABLE_MODELS[self.config["model_type"]],
                    "messages": messages,
                    "max_tokens": self.config.get("max_tokens", 1024),
                    "temperature": self.config.get("temperature", 0.7),
                    "stream": False
                }
                
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status}, {error_text}")
                        raise Exception(f"API error: {response.status}, {error_text}")
                    
                    result = await response.json()
                    
                    # Defensive handling of the response
                    if not result:
                        raise ValueError("Empty response from API")
                    
                    # Handle different response formats
                    if "choices" not in result:
                        logger.error(f"Unexpected response format: {json.dumps(result)}")
                        # Try to extract text from the response directly
                        if "content" in result:
                            text = result["content"]
                        else:
                            text = f"Error: Could not parse response: {str(result)}"
                    else:
                        # Normal response format
                        choices = result.get("choices", [])
                        if not choices:
                            raise ValueError("No choices in response")
                        
                        message = choices[0].get("message", {})
                        text = message.get("content", "No content in response")
                    
                    # Create the response object
                    return ModelResponse(
                        text=text,
                        tokens_used=result.get("usage", {}).get("total_tokens", 0),
                        model_name=self.config.get("model_name", "unknown"),
                        metadata={"raw_response": result}
                    )
                    
        except Exception as e:
                logger.error(f"Error in generate_with_image: {str(e)}")
                return ModelResponse(
                    text=f"Error processing image: {str(e)}",
                    tokens_used=0,
                    model_name=self.config.get("model_name", "unknown"),
                    metadata={"error": str(e)}
                )
    async def chat(self, prompt: str, params: Dict[str, Any] = None) -> str:
            """Generate chat completion using OpenRouter API."""
            if not self.api_key:
                return {"error": "OpenRouter API key not configured"}
                
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": params.get("max_tokens", self.max_tokens),
                    "temperature": params.get("temperature", self.temperature),
                    "top_p": params.get("top_p", self.top_p)
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, headers=headers, json=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            return {"error": f"API Error: {error_text}"}
                            
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                        return {"error": "No response from model"}
                        
            except Exception as e:
                return {"error": f"Chat error: {str(e)}"}