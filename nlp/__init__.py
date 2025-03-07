"""Natural Language Processing modules for the AI Assistant.

This package provides core NLP functionalities including:
- Text summarization
- Sentiment analysis
- Named Entity Recognition (NER)
- Question answering
- Code generation and assistance
"""

from nlp.summarization import Summarizer
from nlp.sentiment import SentimentAnalyzer
from nlp.ner import NamedEntityRecognizer
from nlp.qa import QuestionAnswerer
from nlp.code_assistant import CodeAssistant

__all__ = [
    'Summarizer',
    'SentimentAnalyzer', 
    'NamedEntityRecognizer',
    'QuestionAnswerer',
    'CodeAssistant'
]