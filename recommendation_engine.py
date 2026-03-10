"""
Recommendation engine with optimized algorithms
"""
import numpy as np
import pandas as pd
import heapq
import streamlit as st
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Data class for a single recommendation"""
    game_name: str
    similarity_score: float
    image_url: Optional[str] = None
    app_id: Optional[int] = None


class RecommendationEngine:
    """
    Optimized recommendation engine using content-based filtering
    """
    
    def __init__(
        self,
        game_data: pd.DataFrame,
        similarity_matrix: np.ndarray,
        image_data: pd.DataFrame,
        game_index: Dict[str, int]
    ):
        """
        Initialize recommendation engine
        
        Args:
            game_data: DataFrame with game metadata
            similarity_matrix: Precomputed cosine similarity matrix
            image_data: DataFrame with game image URLs
            game_index: Dictionary mapping game names to indices
        """
        self.game_data = game_data
        self.similarity_matrix = similarity_matrix
        self.image_data = image_data
        self.game_index = game_index
        
        # Build AppID to image URL mapping for O(1) lookup
        self.image_map = dict(zip(
            image_data['AppID'],
            image_data['Headerimage']
        )) if not image_data.empty else {}
        
        logger.info(f"RecommendationEngine initialized with {len(game_data)} games")
    
    def get_game_index(self, game_name: str) -> Optional[int]:
        """
        Get index for a game name with O(1) lookup
        
        Args:
            game_name: Name of the game
            
        Returns:
            Index of the game or None if not found
        """
        return self.game_index.get(game_name)
    
    def _get_top_k_similar(
        self,
        game_idx: int,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most similar games using heap (optimized)
        
        Args:
            game_idx: Index of the query game
            k: Number of recommendations
            
        Returns:
            List of (index, similarity_score) tuples
            
        Complexity: O(n + k*log(k)) instead of O(n*log(n)) with full sort
        """
        similarities = enumerate(self.similarity_matrix[game_idx])
        
        # Use heapq.nlargest for efficient top-k selection
        # This is O(n + k*log(k)) vs O(n*log(n)) for full sort
        top_k = heapq.nlargest(
            k + 1,  # +1 to account for the game itself
            similarities,
            key=lambda x: x[1]
        )
        
        # Remove the first item (the game itself, similarity = 1.0)
        return top_k[1:]
    
    def recommend(
        self,
        game_name: str,
        top_n: int = 6
    ) -> List[Recommendation]:
        """
        Get recommendations for a given game
        
        Args:
            game_name: Name of the game to get recommendations for
            top_n: Number of recommendations to return
            
        Returns:
            List of Recommendation objects
            
        Raises:
            ValueError: If game not found in database
        """
        # O(1) lookup using game_index dictionary
        game_idx = self.get_game_index(game_name)
        
        if game_idx is None:
            available_games = len(self.game_index)
            raise ValueError(
                f"Game '{game_name}' not found in database.\n"
                f"Database contains {available_games} games from 2021-2023."
            )
        
        logger.info(f"Getting {top_n} recommendations for '{game_name}'")
        
        # Get top-k similar games (optimized with heap)
        similar_games = self._get_top_k_similar(game_idx, top_n)
        
        # Build recommendation objects
        recommendations = []
        for idx, score in similar_games:
            game_row = self.game_data.iloc[idx]
            app_id = game_row['AppID']
            
            rec = Recommendation(
                game_name=game_row['Name'],
                similarity_score=score,
                image_url=self.image_map.get(app_id),
                app_id=app_id
            )
            recommendations.append(rec)
        
        logger.info(f"Found {len(recommendations)} recommendations")
        return recommendations
    
    @st.cache_data(ttl=3600)
    def get_cached_recommendations(
        _self,
        game_name: str,
        top_n: int = 6
    ) -> List[Recommendation]:
        """
        Cached version of recommend() for frequently requested games
        
        Note: _self is used to prevent hashing the entire object
        """
        return _self.recommend(game_name, top_n)
    
    def get_random_games(self, n: int = 5) -> List[str]:
        """
        Get random game names for suggestions
        
        Args:
            n: Number of random games to return
            
        Returns:
            List of game names
        """
        sample_size = min(n, len(self.game_data))
        random_indices = np.random.choice(
            len(self.game_data),
            size=sample_size,
            replace=False
        )
        return self.game_data.iloc[random_indices]['Name'].tolist()
    
    def search_games(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for games matching a query string
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching game names
        """
        query_lower = query.lower()
        matches = [
            name for name in self.game_index.keys()
            if query_lower in name.lower()
        ]
        return matches[:limit]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the recommendation system
        
        Returns:
            Dictionary with various statistics
        """
        return {
            'total_games': len(self.game_data),
            'similarity_matrix_shape': self.similarity_matrix.shape,
            'games_with_images': len(self.image_map),
            'coverage': len(self.image_map) / len(self.game_data) * 100
        }
