# 🎮 Steam Game Recommendation System

content-based recommendation system for Steam games using machine learning and natural language processing.

https://github.com/user-attachments/assets/ae766122-af48-4b77-937c-17437cfb41a9

## ✨ Features

- **Content-Based Filtering**: Analyzes game tags, descriptions, and genres
- **Optimized Performance**: O(n + k·log k) recommendation algorithm using heaps
- **Smart Caching**: Streamlit cache fo fast repeated queries
- **Error Handling**: Comprehensive error handling and logging
- **Clean Architecture**: Modular, testable, maintainable code
- **Unit Tests**: Full test coverage with pytest

## Quick Start

### Prerequisites

```bash
python >= 3.8
pip
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ashudevcodes/Steam-Game-Recommendation-System.git
cd Steam-Game-Recommendation-System
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (first time only)
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Data Preparation

1. **Prepare your Steam games CSV** with these columns:
   - `AppID`: Unique game identifier
   - `Name`: Game title
   - `About the game`: Game description
   - `Categories`: Game categories
   - `Genres`: Game genres
   - `Tags`: User-generated tags

2. **Run the preprocessing pipeline**:
```python
from preprocessing import SteamGamePreprocessor

# Initialize preprocessor
preprocessor = SteamGamePreprocessor('data/games.csv')

# Run full pipeline
preprocessor.run_full_pipeline()
```

This will create:
- `data/game_data.pkl`: Processed game metadata
- `data/similarity.npy`: Precomputed similarity matrix
- `models/vectorizer.pkl`: Fitted CountVectorizer

### Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🏗️ Architecture

### Data Flow

```
User Input → Game Selection
     ↓
Recommendation Engine
     ↓
O(1) Index Lookup (Dictionary)
     ↓
O(n + k·log k) Top-K Selection (Heap)
     ↓
Fetch Game Details & Images
     ↓
Display Results
```

### Key Components

1. **Config Module**: Centralized configuration management
2. **Data Loader**: Cached data loading with error handling
3. **Recommendation Engine**: Optimized recommendation algorithms
4. **Preprocessing Pipeline**: End-to-end data processing
5. **Streamlit App**: User interface with proper UX

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest test -v

# Run with coverage
pytest test --cov=. --cov-report=html
```

## 📊 Algorithm Details

### Content-Based Filtering

1. **Text Preprocessing**:
   - Tokenization
   - Lowercasing
   - Porter Stemming
   - Stopword removal

2. **Vectorization**:
   - CountVectorizer (Bag of Words)
   - Max features: 5000
   - Stop words: English

3. **Similarity Computation**:
   - Cosine Similarity
   - Formula: `sim(A,B) = (A·B) / (||A|| × ||B||)`
   - Range: [0, 1] for count vectors

4. **Recommendation**:
   - Top-K selection using heaps
   - Excludes input game
   - Returns games with highest similarity

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t steam-recommender .
docker run -p 8501:8501 steam-recommender
```

## 📝 License

[MIT License](LICENSE) - see LICENSE file for details

## 👤 Author

**Ashish Prasad**
- Email: ashishprasad@gmail.com

## 🙏 Acknowledgments

- Steam for game data
- Scikit-learn for ML tools
- Streamlit for the amazing framework
- The open-source community

---
