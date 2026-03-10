"""
Data loading and caching module
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and caching of all data files"""
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner="Loading game data...")
    def load_game_data() -> pd.DataFrame:
        """
        Load game metadata DataFrame
        
        Returns:
            pd.DataFrame: Game data with columns [AppID, Name, tags]
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            Exception: For other loading errors
        """
        try:
            file_path = config.get_data_path(config.GAME_DATA_FILE)
            logger.info(f"Loading game data from {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Game data file not found: {file_path}\n"
                    f"Please run the preprocessing notebook first."
                )
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded {len(data)} games")
            return data
            
        except Exception as e:
            logger.error(f"Error loading game data: {e}")
            raise
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner="Loading similarity matrix...")
    def load_similarity_matrix() -> np.ndarray:
        """
        Load precomputed similarity matrix
        
        Returns:
            np.ndarray: Similarity matrix of shape (n_games, n_games)
            
        Raises:
            FileNotFoundError: If similarity file doesn't exist
        """
        try:
            file_path = config.get_data_path(config.SIMILARITY_FILE)
            logger.info(f"Loading similarity matrix from {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Similarity matrix not found: {file_path}\n"
                    f"Please run the preprocessing notebook first."
                )
            
            similarity = np.load(file_path)
            logger.info(f"Loaded similarity matrix: {similarity.shape}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error loading similarity matrix: {e}")
            raise
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner="Loading game images...")
    def load_image_data() -> pd.DataFrame:
        """
        Load game image URLs
        
        Returns:
            pd.DataFrame: Image data with columns [AppID, Headerimage]
        """
        try:
            file_path = config.get_data_path(config.IMAGE_DATA_FILE)
            logger.info(f"Loading image data from {file_path}")
            
            if not file_path.exists():
                logger.warning(f"Image data file not found: {file_path}")
                return pd.DataFrame(columns=['AppID', 'Headerimage'])
            
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} game images")
            return data
            
        except Exception as e:
            logger.error(f"Error loading image data: {e}")
            return pd.DataFrame(columns=['AppID', 'Headerimage'])
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL)
    def build_game_index(game_data: pd.DataFrame) -> Dict[str, int]:
        """
        Build name->index lookup dictionary for O(1) access
        
        Args:
            game_data: DataFrame with 'Name' column
            
        Returns:
            Dict mapping game names to indices
        """
        logger.info("Building game name index")
        game_index = {
            name: idx 
            for idx, name in enumerate(game_data['Name'])
        }
        logger.info(f"Built index with {len(game_index)} games")
        return game_index
    
    @classmethod
    def load_all_data(cls) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, int]]:
        """
        Load all required data files
        
        Returns:
            Tuple of (game_data, similarity_matrix, image_data, game_index)
        """
        try:
            game_data = cls.load_game_data()
            similarity = cls.load_similarity_matrix()
            image_data = cls.load_image_data()
            game_index = cls.build_game_index(game_data)
            
            logger.info("All data loaded successfully")
            return game_data, similarity, image_data, game_index
            
        except FileNotFoundError as e:
            logger.error(f"Required data files missing: {e}")
            st.error(str(e))
            st.stop()
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            st.error(f"Error loading data: {e}")
            st.stop()
