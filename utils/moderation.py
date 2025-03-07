"""Content moderation utilities for the AI Assistant."""

import re
from typing import Dict, List, Any, Tuple
from loguru import logger

class ContentModerator:
    """Handles content moderation to prevent harmful or biased outputs."""
    
    def __init__(self):
        """Initialize the content moderator."""
        # Define categories of potentially harmful content
        self.harmful_categories = {
            "hate_speech": [
                r"\b(hate|hateful|hating)\b.*\b(group|community|people|race|gender|religion)\b",
                r"\b(racist|sexist|homophobic|transphobic|bigoted)\b",
            ],
            "violence": [
                r"\b(kill|killing|murder|attack|hurt|harm|injure)\b.*\b(people|person|individual|group)\b",
                r"\b(bomb|shooting|terrorist|attack|violence)\b",
            ],
            "self_harm": [
                r"\b(suicide|self-harm|kill\s+myself|hurt\s+myself|end\s+my\s+life)\b",
                r"\b(ways\s+to\s+die|how\s+to\s+commit)\b",
            ],
            "sexual_content": [
                r"\b(explicit|pornographic|sexual\s+content)\b",
                r"\b(child\s+pornography|underage|minor)\b.*\b(sexual|explicit)\b",
            ],
            "personal_info": [
                r"\b(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})\b",  # Phone numbers
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP addresses
            ],
            "illegal_activity": [
                r"\b(hack|hacking|crack|cracking)\b.*\b(password|account|system)\b",
                r"\b(steal|download|pirate)\b.*\b(movie|music|software|content)\b",
                r"\b(buy|sell|purchase)\b.*\b(drugs|illegal|controlled\s+substance)\b",
            ],
        }
        
        # Define refusal messages for each category
        self.refusal_messages = {
            "hate_speech": "I cannot provide content that promotes hate speech or discrimination.",
            "violence": "I cannot provide content that promotes or glorifies violence.",
            "self_harm": "I cannot provide information that might encourage self-harm. If you're feeling distressed, please reach out to a mental health professional or a crisis helpline.",
            "sexual_content": "I cannot provide explicit or inappropriate sexual content.",
            "personal_info": "I cannot assist with sharing or collecting personal information.",
            "illegal_activity": "I cannot provide assistance with illegal activities.",
            "general": "I cannot provide the requested content as it may violate ethical guidelines."
        }
        
        logger.info("Initialized Content Moderator")
    
    def check_input(self, text: str) -> Tuple[bool, str, str]:
        """Check if user input contains potentially harmful content.
        
        Args:
            text (str): User input text
            
        Returns:
            Tuple[bool, str, str]: (is_harmful, category, message)
        """
        if not text:
            return False, "", ""
        
        for category, patterns in self.harmful_categories.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"Detected potentially harmful content in category: {category}")
                    return True, category, self.refusal_messages.get(category, self.refusal_messages["general"])
        
        return False, "", ""
    
    def check_output(self, text: str) -> Tuple[bool, str, str]:
        """Check if AI output contains potentially harmful content.
        
        Args:
            text (str): AI output text
            
        Returns:
            Tuple[bool, str, str]: (is_harmful, category, message)
        """
        # Similar to check_input but can have different thresholds or rules
        return self.check_input(text)
    
    def sanitize_output(self, text: str) -> str:
        """Sanitize AI output to remove potentially harmful content.
        
        Args:
            text (str): AI output text
            
        Returns:
            str: Sanitized text
        """
        # Check if the text is harmful
        is_harmful, category, message = self.check_output(text)
        
        if is_harmful:
            # Return a sanitized version or the refusal message
            return f"I apologize, but I cannot provide the requested information. {message}"
        
        return text