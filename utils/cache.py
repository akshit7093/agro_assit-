"""Caching module for the AI Assistant."""

from typing import Any, Optional
from loguru import logger

class CacheManager:
    """Handles caching of responses and intermediate results."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self.cache = {}
        logger.info("Initialized CacheManager")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached value if exists, None otherwise
        """
        try:
            return self.cache.get(key)
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
        """
        try:
            self.cache[key] = value
            logger.debug(f"Cached value for key: {key}")
            
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear all cached values."""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise