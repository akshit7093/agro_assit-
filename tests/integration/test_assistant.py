"""Integration tests for the AI Assistant."""

import unittest
import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import AIAssistant

class TestAIAssistant(unittest.TestCase):
    """Integration tests for the AIAssistant class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        cls.assistant = asyncio.run(cls.async_setup())
    
    @staticmethod
    async def async_setup():
        """Async setup for the assistant."""
        return AIAssistant()
    
    def test_chat(self):
        """Test basic chat functionality."""
        response = asyncio.run(self.assistant.chat(
            "Hello, how are you?",
            {"model_name": "deepseek"}  # Using a free model
        ))
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_process_text(self):
        """Test text processing with a specific task."""
        response = asyncio.run(self.assistant.process_text(
            "The quick brown fox jumps over the lazy dog.",
            "summarize",
            {"model_name": "deepseek"}
        ))
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, dict)
        self.assertIn("result", response)

if __name__ == "__main__":
    unittest.main()