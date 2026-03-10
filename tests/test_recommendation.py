"""
Unit tests for the recommendation system
Run with: pytest test_recommendation.py -v
"""
import unittest
import numpy as np
import pandas as pd
from recommendation_engine import RecommendationEngine, Recommendation


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for RecommendationEngine"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        # Create sample game data
        cls.game_data = pd.DataFrame({
            'AppID': [1, 2, 3, 4, 5],
            'Name': ['Game A', 'Game B', 'Game C', 'Game D', 'Game E'],
            'tags': ['action shooter', 'action rpg', 'puzzle casual', 
                     'shooter fps', 'rpg fantasy']
        })
        
        # Create sample similarity matrix
        # Game A (action shooter) should be similar to Game D (shooter fps)
        cls.similarity = np.array([
            [1.0, 0.5, 0.1, 0.8, 0.3],  # Game A
            [0.5, 1.0, 0.2, 0.4, 0.7],  # Game B
            [0.1, 0.2, 1.0, 0.1, 0.2],  # Game C
            [0.8, 0.4, 0.1, 1.0, 0.3],  # Game D
            [0.3, 0.7, 0.2, 0.3, 1.0],  # Game E
        ])
        
        # Create sample image data
        cls.image_data = pd.DataFrame({
            'AppID': [1, 2, 3, 4, 5],
            'Headerimage': [
                'http://image1.jpg',
                'http://image2.jpg',
                'http://image3.jpg',
                'http://image4.jpg',
                'http://image5.jpg'
            ]
        })
        
        # Create game index
        cls.game_index = {
            'Game A': 0,
            'Game B': 1,
            'Game C': 2,
            'Game D': 3,
            'Game E': 4
        }
        
        # Initialize engine
        cls.engine = RecommendationEngine(
            game_data=cls.game_data,
            similarity_matrix=cls.similarity,
            image_data=cls.image_data,
            game_index=cls.game_index
        )
    
    def test_initialization(self):
        """Test that engine initializes correctly"""
        self.assertEqual(len(self.engine.game_data), 5)
        self.assertEqual(self.engine.similarity_matrix.shape, (5, 5))
        self.assertEqual(len(self.engine.game_index), 5)
        self.assertEqual(len(self.engine.image_map), 5)
    
    def test_get_game_index(self):
        """Test game index lookup"""
        self.assertEqual(self.engine.get_game_index('Game A'), 0)
        self.assertEqual(self.engine.get_game_index('Game E'), 4)
        self.assertIsNone(self.engine.get_game_index('Nonexistent Game'))
    
    def test_recommend_returns_correct_count(self):
        """Test that recommend returns requested number of games"""
        recs = self.engine.recommend('Game A', top_n=3)
        self.assertEqual(len(recs), 3)
    
    def test_recommend_excludes_input_game(self):
        """Test that input game is not in recommendations"""
        recs = self.engine.recommend('Game A', top_n=3)
        rec_names = [r.game_name for r in recs]
        self.assertNotIn('Game A', rec_names)
    
    def test_recommend_similarity_order(self):
        """Test that recommendations are ordered by similarity"""
        recs = self.engine.recommend('Game A', top_n=4)
        
        # Game D should be most similar to Game A (similarity = 0.8)
        self.assertEqual(recs[0].game_name, 'Game D')
        self.assertAlmostEqual(recs[0].similarity_score, 0.8, places=5)
        
        # Verify descending order
        for i in range(len(recs) - 1):
            self.assertGreaterEqual(
                recs[i].similarity_score,
                recs[i+1].similarity_score,
                msg="Recommendations not in descending similarity order"
            )
    
    def test_recommend_invalid_game(self):
        """Test that ValueError is raised for invalid game"""
        with self.assertRaises(ValueError) as context:
            self.engine.recommend('Nonexistent Game')
        
        self.assertIn('not found', str(context.exception))
    
    def test_recommendation_has_image_url(self):
        """Test that recommendations include image URLs"""
        recs = self.engine.recommend('Game A', top_n=2)
        
        for rec in recs:
            self.assertIsNotNone(rec.image_url)
            self.assertTrue(rec.image_url.startswith('http'))
    
    def test_recommendation_has_app_id(self):
        """Test that recommendations include AppID"""
        recs = self.engine.recommend('Game A', top_n=2)
        
        for rec in recs:
            self.assertIsNotNone(rec.app_id)
            self.assertIn(rec.app_id, self.game_data['AppID'].values)
    
    def test_get_random_games(self):
        """Test random game selection"""
        random_games = self.engine.get_random_games(n=3)
        
        self.assertEqual(len(random_games), 3)
        self.assertTrue(all(game in self.game_index for game in random_games))
    
    def test_search_games(self):
        """Test game search functionality"""
        # Search for games containing 'Game'
        results = self.engine.search_games('Game', limit=10)
        self.assertEqual(len(results), 5)
        
        # Search for specific game
        results = self.engine.search_games('Game A', limit=10)
        self.assertIn('Game A', results)
        
        # Case-insensitive search
        results = self.engine.search_games('game a', limit=10)
        self.assertIn('Game A', results)
    
    def test_get_statistics(self):
        """Test statistics generation"""
        stats = self.engine.get_statistics()
        
        self.assertEqual(stats['total_games'], 5)
        self.assertEqual(stats['similarity_matrix_shape'], (5, 5))
        self.assertEqual(stats['games_with_images'], 5)
        self.assertEqual(stats['coverage'], 100.0)
    
    def test_similarity_matrix_properties(self):
        """Test mathematical properties of similarity matrix"""
        # Diagonal should be 1.0 (self-similarity)
        for i in range(len(self.similarity)):
            self.assertAlmostEqual(self.similarity[i][i], 1.0, places=5)
        
        # Matrix should be symmetric
        for i in range(len(self.similarity)):
            for j in range(i+1, len(self.similarity)):
                self.assertAlmostEqual(
                    self.similarity[i][j],
                    self.similarity[j][i],
                    places=5,
                    msg=f"Similarity matrix not symmetric at ({i},{j})"
                )
        
        # All values should be between -1 and 1
        self.assertTrue(np.all(self.similarity >= -1.0))
        self.assertTrue(np.all(self.similarity <= 1.0))
    
    def test_heap_optimization(self):
        """Test that top-k selection is efficient"""
        # This is more of a performance test
        # Just verify it produces correct results
        game_idx = 0
        top_k = self.engine._get_top_k_similar(game_idx, k=3)
        
        # Should return 3 results
        self.assertEqual(len(top_k), 3)
        
        # Should be tuples of (index, score)
        for idx, score in top_k:
            self.assertIsInstance(idx, (int, np.integer))
            self.assertIsInstance(score, (float, np.floating))
            
        # Scores should be in descending order
        scores = [score for _, score in top_k]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestRecommendationDataClass(unittest.TestCase):
    """Test the Recommendation data class"""
    
    def test_recommendation_creation(self):
        """Test creating Recommendation objects"""
        rec = Recommendation(
            game_name="Test Game",
            similarity_score=0.85,
            image_url="http://test.jpg",
            app_id=123
        )
        
        self.assertEqual(rec.game_name, "Test Game")
        self.assertEqual(rec.similarity_score, 0.85)
        self.assertEqual(rec.image_url, "http://test.jpg")
        self.assertEqual(rec.app_id, 123)
    
    def test_recommendation_optional_fields(self):
        """Test that optional fields default to None"""
        rec = Recommendation(
            game_name="Test Game",
            similarity_score=0.85
        )
        
        self.assertIsNone(rec.image_url)
        self.assertIsNone(rec.app_id)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
