import os
import shutil
from loguru import logger

def clean_temp_directory(temp_dir: str) -> None:
    """
    Clean up temporary directory by removing all files and subdirectories.

    Args:
        temp_dir (str): Path to the temporary directory
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning temporary directory {temp_dir}: {str(e)}")
        raise