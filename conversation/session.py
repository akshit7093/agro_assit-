"""Session management module for conversation handling."""

import os
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger
from datetime import datetime
from models.openrouter import OpenRouterModel
from models.factory import ModelFactory
from rag.document_store import DocumentStore  # Add this import
from rag.retriever import DocumentRetriever  # Add this import

class SessionManager:
    """Handles conversation session management."""
    
    def __init__(self, document_store=None, retriever=None):
        """Initialize the session manager.
        
        Args:
            document_store: Optional shared document store
            retriever: Optional shared retriever
        """
        self.sessions = {}
        self.current_session_id = None
        self.document_store = document_store
        self.retriever = retriever
        logger.info("Initialized SessionManager")
    
    def get_or_create_session(self, session_id: Optional[str] = None, model_type: str = "gemini") -> 'Session':
        """Get or create session with model type support."""
        try:
            # If we have an existing session and just want to change the model
            if session_id and session_id in self.sessions:
                existing_session = self.sessions[session_id]
                # If model type is different, update the model but keep the history
                if model_type and existing_session.model_type != model_type:
                    # Save the existing messages and files
                    existing_messages = existing_session.messages
                    existing_files = existing_session.files
                    
                    # Create a new session with the new model type but same session_id
                    # Avoid recursive call to get_or_create_session here
                    new_session = Session(session_id=session_id, model_type=model_type)
                    
                    # Restore the messages and files
                    new_session.messages = existing_messages
                    new_session.files = existing_files
                    
                    # Replace the old session
                    self.sessions[session_id] = new_session
                    logger.info(f"Updated session {session_id} with new model type: {model_type}")
                    return new_session
                return existing_session
            
            # Create new session with the specified model type and shared RAG components
            session_id = session_id or str(len(self.sessions) + 1)
            new_session = Session(
                session_id=session_id, 
                model_type=model_type,
                document_store=self.document_store,
                retriever=self.retriever
            )
            self.sessions[session_id] = new_session
            self.current_session_id = session_id
            
            logger.info(f"Created new session: {session_id}")
            return new_session
        except Exception as e:
            logger.error(f"Error managing session: {str(e)}")
            raise

class Session:
    """Represents a conversation session."""
    
    def __init__(self, session_id: Optional[str] = None, model_type: str = "gemini", 
                 document_store=None, retriever=None):
        """Initialize a new session with specific model."""
        self.session_id = session_id or f"session_{id(self)}"
        
        # Validate model type against available OpenRouter models
        if model_type not in ["gemini", "deepseek", "qwen", "NVIDIA"]:
            model_type = "gemini"  # Default to gemini if invalid model type
            logger.warning(f"Invalid model type specified, defaulting to {model_type}")
        
        self.model_type = model_type
        self.messages = []
        self.files = {}  # Store file references
        self.history = []
        self.context = {}
        
        # Use shared RAG components if provided, otherwise create new ones
        self.document_store = document_store or DocumentStore()
        self.retriever = retriever or DocumentRetriever(self.document_store)
        
        # Initialize the model factory
        self.model_factory = ModelFactory()
        
        # Initialize and validate model
        self.llm = self.model_factory.get_model(model_type)
        if not self.llm:
            raise ValueError(f"Failed to initialize model: {model_type}")
        
        logger.info(f"Created new session: {self.session_id} with model: {model_type}")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session history and document store."""
        message = {"role": role, "content": content}
        self.messages.append(message)
        
        # Add to document store for RAG
        self.document_store.add_text(
            content,
            metadata={
                "role": role,
                "timestamp": str(datetime.now()),
                "session_id": self.session_id
            }
        )
        logger.debug(f"Added {role} message to session {self.session_id}")

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using the configured model."""
        try:
            # Get the model from the model factory
            model_name = kwargs.get("model_name", self.model_type)
            try:
                model = self.model_factory.get_model(model_name)
                if model is None:
                    raise ValueError(f"Failed to initialize model {model_name}")
            except Exception as model_init_error:
                logger.error(f"Error initializing model {model_name}: {str(model_init_error)}")
                return f"I'm sorry, but I encountered an error initializing the {model_name} model. Please try a different model or contact support."
            
            # Check if this is a vision request (has image_data)
            is_vision_request = "image_data" in kwargs or kwargs.get("is_vision", False)
            
            # Generate response
            try:
                response = await model.generate(prompt, **kwargs)
            except Exception as model_error:
                error_str = str(model_error)
                
                # Check if this is a rate limit error
                if "429" in error_str or "Quota exceeded" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"Rate limit hit for {model_name}, trying fallback model")
                    
                    # Define fallback models based on whether this is a vision request
                    if is_vision_request:
                        # Vision-capable fallback models only
                        vision_fallbacks = {
                            "gemini": "qwen",     # If Gemini fails, try Qwen (vision capable)
                            "qwen": "gemini",     # If Qwen fails, try Gemini (vision capable)
                            # Add other vision models as they become available
                        }
                        fallback_model_name = vision_fallbacks.get(model_name)
                        
                        # If no appropriate vision fallback, explain the issue
                        if not fallback_model_name:
                            return "I'm sorry, but I've hit a rate limit with the current vision model, and no alternative vision models are currently available. Please try again in a minute."
                    else:
                        # Text-only fallbacks can use any model
                        text_fallbacks = {
                            "gemini": "deepseek",  # If Gemini fails, try DeepSeek
                            "deepseek": "qwen",    # If DeepSeek fails, try Qwen
                            "qwen": "llama",       # If Qwen fails, try Llama
                            "llama": "gemini"      # If all else fails, circle back
                        }
                        fallback_model_name = text_fallbacks.get(model_name, "deepseek")
                    
                    # Try with fallback model
                    try:
                        logger.info(f"Using fallback model: {fallback_model_name}")
                        fallback_model = self.model_factory.get_model(fallback_model_name)
                        response = await fallback_model.generate(prompt, **kwargs)
                        return f"(Note: Used {fallback_model_name} due to rate limits) " + (response.text if hasattr(response, 'text') else str(response))
                    except Exception as fallback_error:
                        # If fallback also fails, raise the original error
                        logger.error(f"Fallback model also failed: {str(fallback_error)}")
                        raise model_error
                else:
                    # If it's not a rate limit error, re-raise
                    raise
            
            # Check if response is an error message
            if isinstance(response, str) and response.startswith("Error:"):
                logger.error(f"Error generating response: {response}")
                return f"I'm sorry, I encountered an issue: {response}"
            
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."

    def add_file(self, file_path: str, file_name: str) -> None:
        """Add a file reference to the session."""
        # Check file size before adding
        try:
            file_size = os.path.getsize(file_path)
            # Set a reasonable file size limit (e.g., 10MB)
            max_file_size = 10 * 1024 * 1024  # 10MB in bytes
            
            if file_size > max_file_size:
                logger.warning(f"File {file_name} is too large ({file_size} bytes). Maximum allowed size is {max_file_size} bytes.")
                raise ValueError(f"File must be {max_file_size/1024/1024:.1f}MB or smaller.")
            
            self.files[file_name] = file_path
            logger.info(f"Added file {file_name} to session {self.session_id}")
            
            # Process the file with the document store
            asyncio.create_task(self.document_store.process_file(file_path))
        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
            raise
        except Exception as e:
            logger.error(f"Error adding file: {str(e)}")
            raise

    def get_file(self, file_name: str) -> Optional[str]:
        """Get file path by name.
        
        Args:
            file_name (str): Name of the file
            
        Returns:
            Optional[str]: Path to the file if found, None otherwise
        """
        return self.files.get(file_name)
    
    def get_all_files(self) -> Dict[str, str]:
        """Get all files in this session.
        
        Returns:
            Dict[str, str]: Dictionary of file names to file paths
        """
        return self.files
    
    async def generate_response(self, message: str, context: Dict[str, Any] = None, relevant_docs: List[Dict[str, Any]] = None) -> str:
        """Generate a response to the user message.
        
        Args:
            message (str): User message
            context (Dict[str, Any], optional): Additional context
            relevant_docs (List[Dict[str, Any]], optional): Relevant documents
            
        Returns:
            str: Generated response
        """
        try:
            # Add user message to history
            self.add_message("user", message)
            
            # Build prompt with conversation history and relevant documents
            prompt = self._build_prompt(message, relevant_docs)
            
            # Generate response
            # Validate model instance before generation
            if not self.llm:
                raise ValueError("Language model not initialized for this session")
            
            response = await self.llm.generate(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Add assistant response to history
            self.add_message("assistant", response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while processing your request. Please try again."

    def _build_prompt(self, message: str, relevant_docs: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build a prompt with conversation history and relevant documents."""
        # Include conversation history
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages[-5:]])
        
        # Include relevant documents if available
        context_str = ""
        if relevant_docs and len(relevant_docs) > 0:
            context_str = "\nRelevant information from your documents:\n" + "\n".join([
                f"Document {i+1}: {doc.get('content', '')[:500]}..." 
                for i, doc in enumerate(relevant_docs[:3])
            ])
            
            # Add file names if available
            file_names = []
            for doc in relevant_docs:
                if 'metadata' in doc and 'file_name' in doc['metadata']:
                    file_names.append(doc['metadata']['file_name'])
            
            if file_names:
                context_str += f"\n\nThese excerpts are from: {', '.join(file_names)}"
        
        # Build the final prompt - modified to be less document-focused
        if context_str:
            prompt = f"""You are a helpful AI assistant. The user has uploaded some documents that you can reference.
    
            Conversation history:
            {history_str}
    
            {context_str}
    
            User's current message: {message}
    
            Please respond to the user's message. If the user's query relates to the documents, use the document information to provide an accurate response. If not, just respond normally as a helpful assistant."""
        else:
            # Simplified prompt when no documents are available
            prompt = f"""You are a helpful AI assistant.
    
            Conversation history:
            {history_str}
    
            User's current message: {message}
    
            Please respond to the user's message as a helpful assistant. You don't need to reference any documents."""
        
        return prompt