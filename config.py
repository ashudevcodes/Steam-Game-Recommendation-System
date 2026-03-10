"""
Configuration management for Steam Game Recommendation System
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    """Application configuration"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / 'data'
    MODELS_DIR: Path = BASE_DIR / 'models'
    LOGS_DIR: Path = BASE_DIR / 'logs'
    
    # Data files
    GAME_DATA_FILE: str = 'game_data.pkl'
    SIMILARITY_FILE: str = 'similariy.npy'  # Changed to .npy for efficiency
    IMAGE_DATA_FILE: str = 'image_data.csv'
    GAME_INDEX_FILE: str = 'game_index.pkl'
    
    # Model parameters
    MAX_FEATURES: int = 5000
    STOP_WORDS: str = 'english'
    TOP_N_RECOMMENDATIONS: int = 6
    
    # Performance
    USE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    # Streamlit settings
    PAGE_TITLE: str = "Steam Games Recommender"
    PAGE_ICON: str = "🎮"
    LAYOUT: str = "wide"
    
    # Contact information
    CONTACT_INFO: dict = None
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        
        # Default contact info
        if self.CONTACT_INFO is None:
            self.CONTACT_INFO = {
                'instagram': 'https://www.instagram.com/ashishprasad__/',
                'linkedin': 'https://www.linkedin.com/in/ashish-prasad-92223a228/',
                'email': 'ashishprasad@gmail.com'
            }
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path for data file"""
        return self.DATA_DIR / filename
    
    def get_model_path(self, filename: str) -> Path:
        """Get full path for model file"""
        return self.MODELS_DIR / filename


# Global config instance
config = Config()
