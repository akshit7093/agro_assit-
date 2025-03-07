"""Data ingestion pipeline for processing various input formats."""

from typing import Dict, List, Union
import pandas as pd
import json
from loguru import logger

class DataIngestionPipeline:
    """Handles ingestion of various data formats and preprocessing."""
    
    def __init__(self):
        """Initialize the data ingestion pipeline."""
        logger.info("Initializing data ingestion pipeline")
    
    def process_csv(self, file_path: str) -> pd.DataFrame:
        """Process CSV files.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Processed data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully processed CSV file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            raise
    
    def process_json(self, file_path: str) -> Dict:
        """Process JSON files.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            Dict: Processed data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully processed JSON file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {str(e)}")
            raise
    
    def process_text(self, file_path: str) -> str:
        """Process text files.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: Processed text content
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            logger.info(f"Successfully processed text file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise
    
    def process_batch(self, file_paths: List[str]) -> List[Union[pd.DataFrame, Dict, str]]:
        """Process multiple files in batch.
        
        Args:
            file_paths (List[str]): List of file paths to process
            
        Returns:
            List[Union[pd.DataFrame, Dict, str]]: List of processed data
        """
        results = []
        for file_path in file_paths:
            try:
                if file_path.endswith('.csv'):
                    results.append(self.process_csv(file_path))
                elif file_path.endswith('.json'):
                    results.append(self.process_json(file_path))
                elif file_path.endswith('.txt'):
                    results.append(self.process_text(file_path))
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        return results