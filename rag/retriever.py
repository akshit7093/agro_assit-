"""Document retrieval module for RAG functionality."""

from typing import List, Dict, Any
from loguru import logger
from rag.document_store import DocumentStore

class DocumentRetriever:
    """Handles document retrieval from the document store."""
    
    def __init__(self, document_store: DocumentStore):
        """Initialize the document retriever.
        
        Args:
            document_store (DocumentStore): Document storage instance
        """
        self.document_store = document_store
        logger.info("Initialized DocumentRetriever")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the given query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents
        """
        try:
            # Use document store's search functionality
            results = self.document_store.search(query, top_k=top_k)
            
            # If no results found, return all documents as fallback
            if not results and len(self.document_store.documents) > 0:
                logger.info("No specific matches found, returning all documents")
                results = [
                    {
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': 0.5  # Default relevance score
                    }
                    for doc_id, doc in self.document_store.documents.items()
                ][:top_k]
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []  # Return empty list instead of raising exception