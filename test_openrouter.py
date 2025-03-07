"""Test script for OpenRouter model implementation."""

import asyncio
import os
from dotenv import load_dotenv
from loguru import logger
from models.openrouter import OpenRouterModel

# Configure logging
logger.add("logs/test.log", rotation="10 MB")

async def test_model(model_type: str, prompt: str):
    """Test a specific model with the given prompt.
    
    Args:
        model_type (str): The model type to test (gemini, deepseek, qwen, llama)
        prompt (str): The prompt to send to the model
    """
    try:
        # Load API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            return
        
        # Configure the model
        model_config = {
            "api_key": api_key,
            "model_type": model_type,
            "model_name": f"OpenRouter-{model_type.capitalize()}",
            "max_tokens": 500,
            "temperature": 0.7,
            "site_url": "https://example.com",
            "site_name": "AI Assistant Test"
        }
        
        # Initialize the model
        model = OpenRouterModel(model_config)
        logger.info(f"Testing {model_type} model with prompt: {prompt}")
        
        # Generate response
        response = await model.generate(prompt)
        
        # Print results
        print(f"\n--- {model_type.upper()} MODEL RESPONSE ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {response.text}")
        print(f"Tokens used: {response.tokens_used}")
        print("----------------------------\n")
        
        return response
    
    except Exception as e:
        logger.error(f"Error testing {model_type} model: {str(e)}")
        print(f"Error with {model_type} model: {str(e)}")
        return None

async def test_streaming(model_type: str, prompt: str):
    """Test streaming with a specific model.
    
    Args:
        model_type (str): The model type to test
        prompt (str): The prompt to send to the model
    """
    try:
        # Load API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            return
        
        # Configure the model
        model_config = {
            "api_key": api_key,
            "model_type": model_type,
            "model_name": f"OpenRouter-{model_type.capitalize()}",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        # Initialize the model
        model = OpenRouterModel(model_config)
        logger.info(f"Testing {model_type} streaming with prompt: {prompt}")
        
        print(f"\n--- {model_type.upper()} STREAMING RESPONSE ---")
        print(f"Prompt: {prompt}")
        print("Response: ", end="")
        
        # Stream response
        async for token in model.generate_stream(prompt):
            print(token, end="", flush=True)
        
        print("\n----------------------------\n")
    
    except Exception as e:
        logger.error(f"Error testing {model_type} streaming: {str(e)}")
        print(f"\nError with {model_type} streaming: {str(e)}")

async def test_multimodal(model_type: str, prompt: str, image_url: str):
    """Test multimodal capabilities with a specific model.
    
    Args:
        model_type (str): The model type to test (should be gemini or qwen)
        prompt (str): The prompt to send to the model
        image_url (str): URL of the image to analyze
    """
    if model_type not in ["gemini", "qwen"]:
        print(f"Model {model_type} does not support multimodal inputs")
        return
    
    try:
        # Load API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            return
        
        # Configure the model
        model_config = {
            "api_key": api_key,
            "model_type": model_type,
            "model_name": f"OpenRouter-{model_type.capitalize()}",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        # Initialize the model
        model = OpenRouterModel(model_config)
        logger.info(f"Testing {model_type} multimodal with prompt: {prompt} and image: {image_url}")
        
        # Format prompt with image
        formatted_prompt = model._format_prompt(prompt, image_url=image_url)
        
        # Prepare request payload
        payload = {
            "model": model.model_id,
            "messages": formatted_prompt,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
        }
        
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": model.config.get('site_url', ''),
            "X-Title": model.config.get('site_name', '')
        }
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{model.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                result = await response.json()
        
        # Extract response text
        response_text = result['choices'][0]['message']['content']
        tokens_used = result.get('usage', {}).get('total_tokens', 0)
        
        # Print results
        print(f"\n--- {model_type.upper()} MULTIMODAL RESPONSE ---")
        print(f"Prompt: {prompt}")
        print(f"Image: {image_url}")
        print(f"Response: {response_text}")
        print(f"Tokens used: {tokens_used}")
        print("----------------------------\n")
    
    except Exception as e:
        logger.error(f"Error testing {model_type} multimodal: {str(e)}")
        print(f"Error with {model_type} multimodal: {str(e)}")

async def main():
    """Run all tests."""
    # Load environment variables
    load_dotenv("config/.env")
    
    # Test prompts
    basic_prompt = "What are the three laws of robotics?"
    creative_prompt = "Write a short poem about artificial intelligence."
    coding_prompt = "Write a Python function to check if a string is a palindrome."
    
    # Test each model with basic text generation
    models = ["gemini", "deepseek", "qwen", "llama"]
    for model in models:
        await test_model(model, basic_prompt)
    
    # Test streaming with one model
    await test_streaming("gemini", creative_prompt)
    
    # Test multimodal capabilities (only for supported models)
    image_url = "https://images.unsplash.com/photo-1546776310-eef45dd6d63c?q=80&w=1000"
    await test_multimodal("gemini", "What's in this image? Describe it in detail.", image_url)

if __name__ == "__main__":
    asyncio.run(main())