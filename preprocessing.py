"""
Data preprocessing pipeline for Steam Game Recommendation System

Run this script to prepare data for the recommendation engine.
Improvements over original:
- Proper error handling
- Progress tracking
- Efficient data processing
- Better code organizaion
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import logging
from typing import List
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamGamePreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self, input_csv: str):
        """
        Initialize preprocessor
        
        Args:
            input_csv: Path to raw Steam games CSV file
        """
        self.input_csv = Path(input_csv)
        self.ps = PorterStemmer()
        self.games_data = None
        self.processed_data = None
        self.similarity_matrix = None
        self.vectorizer = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and select relevant columns"""
        logger.info(f"Loading data from {self.input_csv}")
        
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_csv}")
        
        # Load full dataset
        df = pd.read_csv(self.input_csv)
        logger.info(f"Loaded {len(df)} games")
        
        # Select required columns
        required_cols = ['AppID', 'Name', 'About the game', 'Categories', 'Genres', 'Tags']
        
        # Check if all required columns exist
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.games_data = df[required_cols].copy()
        
        # Drop rows with missing critical data
        initial_count = len(self.games_data)
        self.games_data.dropna(subset=['Name', 'Tags'], inplace=True)
        dropped = initial_count - len(self.games_data)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing Name or Tags")
        
        logger.info(f"Selected {len(self.games_data)} games with complete data")
        return self.games_data
    
    def _process_text_column(self, column: pd.Series) -> pd.Series:
        """
        Process a text column: split, remove spaces, convert to list
        
        Args:
            column: Pandas Series with text data
            
        Returns:
            Processed Series with lists of tokens
        """
        # Fill NaN with empty string
        column = column.fillna('')
        
        # Split into words
        column = column.apply(lambda x: x.split() if isinstance(x, str) else [])
        
        # Remove spaces within tokens
        column = column.apply(lambda x: [token.replace(" ", "") for token in x])
        
        return column
    
    def preprocess_features(self) -> pd.DataFrame:
        """Process all text features"""
        logger.info("Processing text features...")
        
        df = self.games_data.copy()
        
        # Process each text column
        text_columns = ['About the game', 'Categories', 'Genres', 'Tags']
        
        for col in tqdm(text_columns, desc="Processing columns"):
            df[col] = self._process_text_column(df[col])
        
        # Combine all features into single 'tags' column
        logger.info("Combining features...")
        df['tags'] = (
            df['About the game'] +
            df['Categories'] +
            df['Genres'] +
            df['Tags']
        )
        
        # Convert list to string
        df['tags'] = df['tags'].apply(lambda x: " ".join(x))
        
        # Lowercase normalization
        df['tags'] = df['tags'].apply(lambda x: x.lower())
        
        # Select final columns
        self.processed_data = df[['AppID', 'Name', 'tags']].copy()
        
        logger.info("Feature processing complete")
        return self.processed_data
    
    def stem_text(self, text: str) -> str:
        """
        Apply Porter stemming to text
        
        Args:
            text: Input text string
            
        Returns:
            Stemmed text
        """
        tokens = text.split()
        stemmed = [self.ps.stem(token) for token in tokens]
        return " ".join(stemmed)
    
    def apply_stemming(self) -> pd.DataFrame:
        """Apply stemming to all tags"""
        logger.info("Applying Porter stemming...")
        
        tqdm.pandas(desc="Stemming")
        self.processed_data['tags'] = self.processed_data['tags'].progress_apply(
            self.stem_text
        )
        
        logger.info("Stemming complete")
        return self.processed_data
    
    def vectorize_text(self) -> np.ndarray:
        """
        Convert text to numerical vectors using CountVectorizer
        
        Returns:
            Sparse matrix of word counts
        """
        logger.info("Vectorizing text with CountVectorizer...")
        
        self.vectorizer = CountVectorizer(
            max_features=config.MAX_FEATURES,
            stop_words=config.STOP_WORDS
        )
        
        vectors = self.vectorizer.fit_transform(self.processed_data['tags'])
        
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Vector shape: {vectors.shape}")
        
        return vectors
    
    def compute_similarity(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix
        
        Args:
            vectors: Sparse matrix of game vectors
            
        Returns:
            Dense similarity matrix
        """
        logger.info("Computing cosine similarity matrix...")
        logger.info("This may take several minutes for large datasets...")
        
        # Compute similarity (automatically handles sparse matrices efficiently)
        self.similarity_matrix = cosine_similarity(vectors, dense_output=True)
        
        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        logger.info(f"Memory usage: ~{self.similarity_matrix.nbytes / 1e9:.2f} GB")
        
        return self.similarity_matrix
    
    def save_data(self):
        """Save processed data and models"""
        logger.info("Saving processed data...")
        
        # Save game data
        game_data_path = config.get_data_path(config.GAME_DATA_FILE)
        with open(game_data_path, 'wb') as f:
            pickle.dump(self.processed_data, f)
        logger.info(f"Saved game data to {game_data_path}")
        
        # Save similarity matrix (use NumPy format for efficiency)
        similarity_path = config.get_data_path(config.SIMILARITY_FILE)
        np.save(similarity_path, self.similarity_matrix)
        logger.info(f"Saved similarity matrix to {similarity_path}")
        
        # Save vectorizer for potential future use
        vectorizer_path = config.get_model_path('vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Saved vectorizer to {vectorizer_path}")
        
        logger.info("All data saved successfully!")
    
    def run_full_pipeline(self):
        """Execute complete preprocessing pipeline"""
        logger.info("=" * 60)
        logger.info("Starting full preprocessing pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Preprocess features
            self.preprocess_features()
            
            # Step 3: Apply stemming
            self.apply_stemming()
            
            # Step 4: Vectorize
            vectors = self.vectorize_text()
            
            # Step 5: Compute similarity
            self.compute_similarity(vectors)
            
            # Step 6: Save everything
            self.save_data()
            
            logger.info("=" * 60)
            logger.info("Preprocessing pipeline completed successfully!")
            logger.info("=" * 60)
            
            # Print summary statistics
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total games processed: {len(self.processed_data)}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        print(f"Sample games:")
        for name in self.processed_data['Name'].head(5):
            print(f"  - {name}")
        print("=" * 60 + "\n")


def test_recommendations(preprocessor: SteamGamePreprocessor, test_game: str = "Stray"):
    """
    Test the recommendation system
    
    Args:
        preprocessor: SteamGamePreprocessor instance
        test_game: Game name to test
    """
    logger.info(f"\nTesting recommendations for '{test_game}'...")
    
    try:
        # Find game index
        idx = preprocessor.processed_data[
            preprocessor.processed_data['Name'] == test_game
        ].index[0]
        
        # Get similarities
        similarities = list(enumerate(preprocessor.similarity_matrix[idx]))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Print top 5 recommendations
        print(f"\nTop 5 recommendations for '{test_game}':")
        for i, (game_idx, score) in enumerate(similarities[1:6], 1):
            game_name = preprocessor.processed_data.iloc[game_idx]['Name']
            print(f"{i}. {game_name} (similarity: {score:.4f})")
            
    except IndexError:
        logger.warning(f"Game '{test_game}' not found in dataset")


if __name__ == "__main__":
    # Example usage
    input_csv = "data/game.csv"
    
    # Initialize and run
    preprocessor = SteamGamePreprocessor(input_csv)
    preprocessor.run_full_pipeline()
    
    # Test recommendations
    test_recommendations(preprocessor, "Stray")
