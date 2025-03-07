"""Retrieval-Augmented Generation (RAG) components for the AI Assistant.

This package provides components for enhancing the AI assistant's responses through
document-based context retrieval. It includes:

- Document Store: Manages document storage, indexing, and vector embeddings
- Document Retriever: Implements semantic search and relevance ranking
- Prompt Builder: Dynamically constructs prompts using retrieved context

The RAG implementation enables the assistant to ground its responses in relevant
documents and knowledge sources, improving accuracy and reliability.
"""

from .document_store import DocumentStore
from .retriever import DocumentRetriever

__all__ = ['DocumentStore', 'DocumentRetriever']