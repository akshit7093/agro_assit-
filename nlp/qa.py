"""Question answering module for the AI Assistant."""

from typing import Dict, Optional, Any, List
from loguru import logger
from models.factory import ModelFactory

class QuestionAnswerer:
    """Handles question answering using LLMs."""
    
    def __init__(self, model_factory: ModelFactory):
        """Initialize the question answerer.
        
        Args:
            model_factory (ModelFactory): Factory for creating LLM instances
        """
        self.model_factory = model_factory
        logger.info("Initialized Question Answerer")
    
    async def answer_question(self, question: str, model_config: Optional[Dict[str, Any]] = None, 
                             context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Answer a question based on optional context.
        
        Args:
            question (str): Question to answer
            model_config (Optional[Dict[str, Any]]): Configuration for the model
            context (Optional[List[str]]): Optional context passages
            
        Returns:
            Dict[str, Any]: Answer and metadata
        """
        try:
            # Get model with the specified configuration
            model_name = model_config.get("model_name") if model_config else None
            model = self.model_factory.get_model(model_name)
            
            # Construct prompt based on whether context is provided
            if context:
                context_text = "\n\n".join(context)
                prompt = f"""Answer the following question based on the provided context:
                
                Context:
                {context_text}
                
                Question: {question}
                
                Answer:"""
            else:
                prompt = f"""Answer the following question to the best of your ability:
                
                Question: {question}
                
                Answer:"""
            
            # Generate answer
            response = await model.generate(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                "question": question,
                "answer": response_text,
                "has_context": bool(context),
                "model_used": model.model_id
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise