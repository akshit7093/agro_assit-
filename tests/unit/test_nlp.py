"""Unit tests for NLP components."""

import unittest
import asyncio
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nlp.summarization import Summarizer
from nlp.sentiment import SentimentAnalyzer
from nlp.ner import NamedEntityRecognizer
from nlp.qa import QuestionAnswerer
from nlp.code_assistant import CodeAssistant
from models.factory import ModelFactory

class TestSummarizer(unittest.TestCase):
    """Test cases for the Summarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_factory = MagicMock(spec=ModelFactory)
        self.model_mock = MagicMock()
        self.model_factory.get_model.return_value = self.model_mock
        self.summarizer = Summarizer(self.model_factory)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.summarizer.model_factory, self.model_factory)
    
    def test_summarize_text(self):
        """Test summarize_text method."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "This is a summary."
        self.model_mock.generate.return_value = asyncio.Future()
        self.model_mock.generate.return_value.set_result(mock_response)
        
        # Run the test
        result = asyncio.run(self.summarizer.summarize_text(
            "This is a long text that needs to be summarized.",
            {"model_name": "test_model"}
        ))
        
        # Assertions
        self.assertIn("summary", result)
        self.assertEqual(result["summary"], "This is a summary.")
        self.model_factory.get_model.assert_called_once()

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for the SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_factory = MagicMock(spec=ModelFactory)
        self.model_mock = MagicMock()
        self.model_factory.get_model.return_value = self.model_mock
        self.sentiment_analyzer = SentimentAnalyzer(self.model_factory)
    
    def test_analyze_sentiment(self):
        """Test analyze_sentiment method."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = '{"sentiment": "positive", "score": 0.8}'
        self.model_mock.generate.return_value = asyncio.Future()
        self.model_mock.generate.return_value.set_result(mock_response)
        
        # Run the test
        result = asyncio.run(self.sentiment_analyzer.analyze_sentiment(
            "I love this product!",
            {"model_name": "test_model"}
        ))
        
        # Assertions
        self.assertIn("sentiment", result)
        self.assertEqual(result["sentiment"], "positive")

if __name__ == "__main__":
    unittest.main()