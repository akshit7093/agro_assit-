"""Performance metrics tracking module for the AI Assistant."""

from typing import Dict, Any
from loguru import logger

class MetricsTracker:
    """Tracks usage metrics for the AI Assistant."""
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics = {
            "processing_count": 0,
            "total_tokens_processed": 0,
            "model_usage": {}
        }
    
    def track_processing(self, task: str, text_length: int):
        """Track a text processing task.
        
        Args:
            task (str): The type of task performed
            text_length (int): Length of the processed text
        """
        self.metrics["processing_count"] += 1
        self.metrics["total_tokens_processed"] += text_length // 4  # Rough estimate of tokens
        
        # Track by task type
        if task not in self.metrics:
            self.metrics[task] = 0
        self.metrics[task] += 1
    
    def track_interaction(self, model: str, prompt_length: int, response_length: int, latency: float):
        """Track a chat interaction with the model.
        
        Args:
            model (str): The model used for the interaction
            prompt_length (int): Length of the user prompt
            response_length (int): Length of the model response
            latency (float): Response time in seconds
        """
        # Track model usage
        if model not in self.metrics["model_usage"]:
            self.metrics["model_usage"][model] = {
                "interactions": 0,
                "total_prompt_length": 0,
                "total_response_length": 0,
                "total_latency": 0
            }
        
        # Update metrics for this model
        self.metrics["model_usage"][model]["interactions"] += 1
        self.metrics["model_usage"][model]["total_prompt_length"] += prompt_length
        self.metrics["model_usage"][model]["total_response_length"] += response_length
        self.metrics["model_usage"][model]["total_latency"] += latency
    
    def get_metrics(self):
        """Get the current metrics.
        
        Returns:
            dict: The current metrics
        """
        return self.metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        try:
            self.metrics = {
                'total_requests': 0,
                'total_tokens': 0,
                'task_counts': {},
                'error_counts': {},
                'average_latency': 0
            }
            logger.info("Reset metrics")
            
        except Exception as e:
            logger.error(f"Error resetting metrics: {str(e)}")
            raise