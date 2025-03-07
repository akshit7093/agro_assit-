"""Document storage and indexing module for RAG functionality."""

import os
import base64
from typing import Dict, List, Any
from loguru import logger
import datetime 
class DocumentStore:
    """Manages document storage and indexing for RAG."""
    
    def __init__(self):
        """Initialize the document store."""
        self.documents = {}  # Dictionary to store documents
        self.index = {}
        logger.info("Initialized DocumentStore")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a document to the store.
        
        Args:
            doc_id (str): Unique document identifier
            content (str): Document content
            metadata (Dict[str, Any], optional): Additional document metadata
        """
        try:
            # Handle video summary data
            if metadata and metadata.get('type') == 'video_summary':
                self.documents[doc_id] = {
                    'content': content,
                    'metadata': metadata,
                    'video_data': metadata.get('video_data', {})
                }
            else:
                self.documents[doc_id] = {
                    'content': content,
                    'metadata': metadata or {}
                }
            logger.info(f"Added document: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def add_text(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add text content as a document.
        
        Args:
            content (str): Text content to add
            metadata (Dict[str, Any], optional): Additional document metadata
        """
        try:
            doc_id = f"doc_{len(self.documents) + 1}"
            self.add_document(doc_id, content, metadata)
        except Exception as e:
            logger.error(f"Error adding text: {str(e)}")
            raise
    
    async def process_file(self, file_path: str) -> None:
        """Process and add a file to the document store.
        
        Args:
            file_path (str): Path to the file
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Check file size
            file_size = os.path.getsize(file_path)
            max_file_size = 10 * 1024 * 1024  # 10MB in bytes
            
            if file_size > max_file_size:
                logger.error(f"File too large: {file_path} ({file_size} bytes)")
                raise ValueError(f"File must be {max_file_size/1024/1024:.1f}MB or smaller.")
            
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            
            # Clear any previous documents with the same file name to avoid confusion
            self._remove_documents_by_filename(file_name)
            
            # Process based on file type
            if file_ext == '.txt':
                # No file size limit for text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    self.add_text(content, metadata={"source": file_path, "type": "text", "file_name": file_name})
            
            elif file_ext == '.pdf':
                try:
                    # Remove size limit for PDF processing
                    from PyPDF2 import PdfReader
                    reader = PdfReader(file_path)
                    content = ""
                    # Process all pages without limit
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                    self.add_text(content, metadata={"source": file_path, "type": "pdf", "file_name": file_name})
                except ImportError:
                    content = f"PDF file: {file_name} (Install PyPDF2 for content extraction)"
                    self.add_text(content, metadata={"source": file_path, "type": "pdf", "file_name": file_name})
            
            elif file_ext == '.csv':
                # Remove size limit for CSV files
                import pandas as pd
                # Use chunksize for large files
                content = ""
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    content += chunk.to_string() + "\n\n"
                self.add_text(content, metadata={"source": file_path, "type": "csv", "file_name": file_name})
            
            elif file_ext == '.json':
                # Remove size limit for JSON files
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
                    self.add_text(content, metadata={"source": file_path, "type": "json", "file_name": file_name})
            
            elif file_ext in ['.docx', '.doc']:
                try:
                    # Remove size limit for DOCX files
                    import docx
                    doc = docx.Document(file_path)
                    content = "\n".join([para.text for para in doc.paragraphs])
                    self.add_text(content, metadata={"source": file_path, "type": "docx", "file_name": file_name})
                except ImportError:
                    content = f"Document file: {file_name} (Install python-docx for content extraction)"
                    self.add_text(content, metadata={"source": file_path, "type": "docx", "file_name": file_name})
            
            # Image support
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                try:
                    # Optional: Use OCR to extract text from images
                    import pytesseract
                    from PIL import Image
                    
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img)
                    
                    # Store both the extracted text and image path
                    self.add_text(text, metadata={
                        "source": file_path, 
                        "type": "image",
                        "image_path": file_path,
                        "file_name": file_name
                    })
                except ImportError:
                    # If OCR libraries aren't available, just store the path
                    self.add_text(f"Image file: {file_name}", metadata={
                        "source": file_path, 
                        "type": "image",
                        "image_path": file_path,
                        "file_name": file_name
                    })
            
            # Video support
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                try:
                    import cv2
                    
                    # Extract basic video metadata
                    video = cv2.VideoCapture(file_path)
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # Extract frames at intervals for potential analysis
                    frames_text = f"Video file: {file_name}\nDuration: {duration:.2f} seconds\nFrames: {frame_count}\nFPS: {fps:.2f}\n\n"
                    
                    if frame_count > 0:
                        # Extract a few frames for analysis
                        interval = max(1, frame_count // 5)  # Get 5 frames max
                        for i in range(0, min(frame_count, 5*interval), interval):
                            video.set(cv2.CAP_PROP_POS_FRAMES, i)
                            success, frame = video.read()
                            if success:
                                frames_text += f"Frame at {i/fps:.2f}s: [Frame data available]\n"
                    
                    video.release()
                    
                    # Store video metadata and any extracted text
                    self.add_text(frames_text, metadata={
                        "source": file_path,
                        "type": "video",
                        "duration": duration,
                        "frame_count": frame_count,
                        "fps": fps,
                        "file_name": file_name
                    })
                    
                except ImportError:
                    # If video processing libraries aren't available
                    self.add_text(f"Video file: {file_name}", metadata={
                        "source": file_path,
                        "type": "video",
                        "video_path": file_path,
                        "file_name": file_name
                    })
            
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                # Store basic info about unsupported files
                self.add_text(f"Unsupported file: {file_name} (type: {file_ext})", metadata={
                    "source": file_path,
                    "type": "unsupported",
                    "file_extension": file_ext,
                    "file_name": file_name
                })
            
            logger.info(f"Added document from {file_path} to the store")
        
        except Exception as e:
            logger.error(f"Error adding document from {file_path}: {str(e)}")
            raise
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve a document from the store.
        
        Args:
            doc_id (str): Document identifier
            
        Returns:
            Dict[str, Any]: Document content and metadata
        """
        try:
            return self.documents.get(doc_id)
            
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            raise
    
    def get_all_documents(self):
        """Get all documents in the store.
        
        Returns:
            dict: All documents in the store
        """
        try:
            return self.documents
        except Exception as e:
            logger.error(f"Error retrieving all documents: {str(e)}")
            return {}

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents
        """
        try:
            # Simple keyword-based search for now
            # Could be enhanced with embeddings/semantic search
            results = []
            for doc_id, doc in self.documents.items():
                content = doc['content'].lower()
                query_terms = query.lower().split()
                
                # Calculate simple relevance score
                score = sum(term in content for term in query_terms) / len(query_terms)
                
                if score > 0:
                    results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': score
                    })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def _remove_documents_by_filename(self, file_name: str) -> None:
        """Remove any documents associated with a specific filename.
        
        Args:
            file_name (str): Name of the file to remove
        """
        docs_to_remove = []
        for doc_id, doc in self.documents.items():
            if doc.get('metadata', {}).get('file_name') == file_name:
                docs_to_remove.append(doc_id)
        
        for doc_id in docs_to_remove:
            del self.documents[doc_id]
            logger.info(f"Removed document {doc_id} associated with {file_name}")