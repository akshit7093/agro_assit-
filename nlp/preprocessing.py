"""Text preprocessing pipeline for NLP tasks."""

from typing import List, Optional
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from loguru import logger

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    """Handles text preprocessing for NLP tasks."""

    def __init__(self):
        """Initialize the text preprocessor."""
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        logger.info("Initialized TextPreprocessor")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text into words.

        Args:
            text (str): Input text

        Returns:
            List[str]: List of tokens
        """
        return nltk.word_tokenize(text)

    def lowercase(self, tokens: List[str]) -> List[str]:
        """Convert tokens to lowercase.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[str]: Lowercased tokens
        """
        return [token.lower() for token in tokens]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[str]: Tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]

    def normalize(self, tokens: List[str], method: str = 'lemmatize') -> List[str]:
        """Normalize tokens using stemming or lemmatization.

        Args:
            tokens (List[str]): List of tokens
            method (str): Normalization method ('stem' or 'lemmatize')

        Returns:
            List[str]: Normalized tokens
        """
        if method == 'stem':
            return [self.stemmer.stem(token) for token in tokens]
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def handle_special_chars(self, tokens: List[str]) -> List[str]:
        """Remove special characters from tokens.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[str]: Cleaned tokens
        """
        return [re.sub(f'[{re.escape(string.punctuation)}]', '', token) for token in tokens]

    def preprocess(self, text: Optional[str]) -> List[str]:
        """Complete text preprocessing pipeline.

        Args:
            text (Optional[str]): Input text

        Returns:
            List[str]: Preprocessed tokens
        """
        if not text:
            return []

        try:
            tokens = self.tokenize(text)
            tokens = self.lowercase(tokens)
            tokens = self.remove_stopwords(tokens)
            tokens = self.normalize(tokens)
            tokens = self.handle_special_chars(tokens)
            tokens = [token for token in tokens if token]  # Remove empty strings
            return tokens
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise