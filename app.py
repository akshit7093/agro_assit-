"""Main application entry point for the Multi-Functional AI Assistant."""

import os
from dotenv import load_dotenv
from loguru import logger
from models.factory import ModelFactory
from processors.ingestion import DataIngestionPipeline
from nlp.summarization import Summarizer
from nlp.sentiment import SentimentAnalyzer
from nlp.ner import NamedEntityRecognizer
from nlp.qa import QuestionAnswerer
from nlp.code_assistant import CodeAssistant
from rag.document_store import DocumentStore
from rag.retriever import DocumentRetriever
from conversation.session import SessionManager
from utils.cache import CacheManager
from utils.metrics import MetricsTracker
from typing import Dict, Any, Optional, List

# Initialize logging
logger.add("logs/app.log", rotation="500 MB")

class AIAssistant:
    """Main AI Assistant class that orchestrates all functionalities."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv(os.path.join("config", ".env"))
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.data_pipeline = DataIngestionPipeline()
        self.cache_manager = CacheManager()
        self.metrics_tracker = MetricsTracker()
        
        # Initialize NLP components
        self.summarizer = Summarizer(self.model_factory)
        self.sentiment_analyzer = SentimentAnalyzer(self.model_factory)
        self.ner = NamedEntityRecognizer(self.model_factory)
        self.qa = QuestionAnswerer(self.model_factory)
        self.code_assistant = CodeAssistant(self.model_factory)
        
        # Initialize RAG components
        self.document_store = DocumentStore()
        self.retriever = DocumentRetriever(self.document_store)
        
        # Initialize session management
        # Initialize session management with RAG components
        self.session_manager = SessionManager(document_store=self.document_store, retriever=self.retriever)
        
        logger.info("AI Assistant initialized successfully")
    
    async def process_text(self, text: str, task: str, model_config: Optional[Dict[str, Any]] = None) -> dict:
        """Process text based on the specified task."""
        try:
            cache_key = f"{task}:{text}"
            if cached_result := self.cache_manager.get(cache_key):
                return cached_result
    
            result = None
            if task == "summarize":
                result = await self.summarizer.summarize(text, model_config)
            elif task == "sentiment":
                result = await self.sentiment_analyzer.analyze(text, model_config)
            elif task == "ner":
                result = await self.ner.extract_entities(text, model_config)
            elif task == "qa":
                result = await self.qa.answer_question(text, model_config)
            elif task == "code":
                result = await self.code_assistant.generate_code(text, model_config)
            elif task == "analyze":
                # Comprehensive analysis combining all features
                result = await self.comprehensive_analysis(text, model_config)
            
            if result:
                self.cache_manager.set(cache_key, result)
                self.metrics_tracker.track_processing(task, len(text))
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text for task {task}: {str(e)}")
            raise
    
    async def comprehensive_analysis(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform a comprehensive analysis using all available AI capabilities.
        
        Args:
            text (str): Text to analyze
            model_config (Optional[Dict[str, Any]]): Configuration for the model
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        try:
            # Track start time for performance metrics
            start_time = time.time()
            
            # Run all analyses in parallel for efficiency
            summary_task = self.summarizer.summarize(text, model_config)
            sentiment_task = self.sentiment_analyzer.analyze(text, model_config)
            ner_task = self.ner.extract_entities(text, model_config)
            
            # Generate questions about the text
            model_name = model_config.get("model_name") if model_config else None
            model = self.model_factory.get_model(model_name)
            if not model:
                raise ValueError(f"Failed to initialize model: {model_name}")
            
            # Generate key questions about the text
            questions_prompt = f"Generate 3 important questions about this text:\n\n{text}"
            questions_response = await model.generate(questions_prompt)
            questions_text = questions_response.text if hasattr(questions_response, 'text') else str(questions_response)
            
            # Extract questions (simple approach - could be more sophisticated)
            questions = [q.strip() for q in questions_text.split('\n') if '?' in q][:3]
            
            # Answer the generated questions
            qa_tasks = [self.qa.answer_question(q, model_config) for q in questions]
            
            # Check for code snippets and analyze if found
            code_analysis = None
            code_blocks = self.extract_code_blocks(text)
            if code_blocks:
                code_analysis_tasks = [self.code_assistant.explain_code(code, model_config) for code in code_blocks]
                code_analysis = await asyncio.gather(*code_analysis_tasks)
            
            # Retrieve relevant documents for context
            relevant_docs = self.retriever.retrieve(text, top_k=3)
            
            # Wait for all analysis tasks to complete
            summary, sentiment, entities = await asyncio.gather(
                summary_task, 
                sentiment_task, 
                ner_task
            )
            
            # Get answers to questions
            answers = await asyncio.gather(*qa_tasks) if questions else []
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Compile final comprehensive analysis
            comprehensive_result = {
                "summary": summary,
                "sentiment": sentiment,
                "entities": entities,
                "questions_and_answers": [{"question": q, "answer": a} for q, a in zip(questions, answers)],
                "code_analysis": code_analysis,
                "relevant_documents": [
                    {"title": doc.get("metadata", {}).get("title", "Unknown"), 
                     "relevance": doc.get("score", 0)} 
                    for doc in relevant_docs
                ],
                "processing_time": processing_time,
                "model_used": model_config.get("model_name", "default") if model_config else "default"
            }
            
            # Generate an executive summary of all findings
            executive_summary_prompt = f"""
            Based on the following analysis of a text, provide a concise executive summary:
            
            Summary: {summary.get('summary', '')}
            Sentiment: {sentiment.get('analysis', '')}
            Key Entities: {entities.get('entities', '')}
            
            Executive Summary:
            """
            
            executive_response = await model.generate(executive_summary_prompt)
            comprehensive_result["executive_summary"] = executive_response.text if hasattr(executive_response, 'text') else str(executive_response)
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text using simple pattern matching.
        
        Args:
            text (str): Text that may contain code blocks
            
        Returns:
            List[str]: Extracted code blocks
        """
        code_blocks = []
        
        # Look for code blocks marked with triple backticks
        import re
        pattern = r"```(?:\w+)?\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            code_blocks.extend(matches)
        
        # Also look for indented code blocks (4+ spaces)
        lines = text.split('\n')
        current_block = []
        in_block = False
        
        for line in lines:
            if line.startswith('    ') and not line.strip() == '':
                if not in_block:
                    in_block = True
                current_block.append(line.lstrip())
            elif in_block:
                if current_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                in_block = False
        
        # Add the last block if there is one
        if current_block:
            code_blocks.append('\n'.join(current_block))
        
        return code_blocks
    
    async def chat(self, message: str, options: Dict[str, Any] = None) -> str:
        """Process a chat message and return a response.
        
        Args:
            message (str): User's message
            options (Dict[str, Any], optional): Additional options. Defaults to None.
            
        Returns:
            str: Assistant's response
        """
        try:
            # Get model name from options or use default
            options = options or {}
            model_name = options.get("model_name", "gemini")
            
            # Verify model availability
            model = self.model_factory.get_model(model_name)
            
            # Get or create a session
            current_session_id = self.session_manager.current_session_id
            session = self.session_manager.get_or_create_session(
                session_id=current_session_id,
                model_type=model_name
            )
            
            # Retrieve relevant documents for RAG if we have documents
            doc_count = len(self.document_store.documents) if hasattr(self, "document_store") else 0
            
            if doc_count > 0:
                relevant_docs = self.retriever.retrieve(message, top_k=3)
                logger.info(f"Retrieved {len(relevant_docs)} documents for query: {message[:50]}...")
            else:
                # No documents available, proceed without RAG
                relevant_docs = []
                logger.info("No documents available in document store, proceeding without RAG")
            
            # Generate response with or without RAG context
            response = await session.generate_response(message, context=options, relevant_docs=relevant_docs)
            
            # Track metrics
            self.metrics_tracker.track_interaction(
                model=model_name,
                prompt_length=len(message),
                response_length=len(response),
                latency=0  # Would track actual latency in production
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"I encountered an error: {str(e)}"

if __name__ == "__main__":
    main()