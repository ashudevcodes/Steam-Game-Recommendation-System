# 🚀 Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Steam-Game-Recommendation-System.git
cd Steam-Game-Recommendation-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run setup
python setup.py

# ✓ Installation complete!
```

---

## Data Preparation (10 minutes)

### Option A: Use Sample Data
```bash
# Download sample dataset
# [Add your dataset download link]

# Place in project directory as 'steam_games.csv'
```

### Option B: Use Your Own Data

Your CSV must have these columns:
- `AppID`: Game ID (integer)
- `Name`: Game title (string)
- `About the game`: Description (string)
- `Categories`: Categories (string)
- `Genres`: Genres (string)
- `Tags`: Tags (string)

---

## Preprocessing (15 minutes)

```python
from preprocessing import SteamGamePreprocessor

# Initialize
preprocessor = SteamGamePreprocessor('steam_games.csv')

# Run pipeline (this will take 10-15 minutes)
preprocessor.run_full_pipeline()

# Output:
# ✓ data/game_data.pkl
# ✓ data/similarity.npy
# ✓ models/vectorizer.pkl
```

---

## Running the App (1 minute)

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

**That's it! You're running! 🎉**

---

## Common Commands

```bash
# Run app
streamlit run app.py

# Run tests
pytest test_recommendation.py -v

# Run tests with coverage
pytest test_recommendation.py --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html

# Stop app
Ctrl + C
```

---

## File Locations

```
Project Root/
├── app.py                    # ← Run this
├── preprocessing.py          # ← Run this first
├── config.py                 # Configuration
├── data_loader.py           # Data loading
├── recommendation_engine.py # Core logic
├── test_recommendation.py   # Tests
├── requirements.txt         # Dependencies
├── setup.py                 # Setup script
│
├── data/                    # ← Generated files go here
│   ├── game_data.pkl
│   ├── similarity.npy
│   └── image_data.csv (optional)
│
├── models/                  # ← Trained models
│   └── vectorizer.pkl
│
└── logs/                    # ← Application logs
    └── app.log
```

---

## Quick Troubleshooting

### "FileNotFoundError: game_data.pkl"
```bash
# Run preprocessing first
python preprocessing.py
```

### "ModuleNotFoundError"
```bash
# Install dependencies
pip install -r requirements.txt
```

### "NLTK data not found"
```python
python -c "import nltk; nltk.download('stopwords')"
```

### App is slow
```bash
# Check if caching is enabled (it should be by default)
# Clear Streamlit cache: C in the app
# Or restart: Ctrl+C then streamlit run app.py
```

---

## Configuration

Edit `config.py` to customize:

```python
# Change number of recommendations
TOP_N_RECOMMENDATIONS: int = 6  # Change to 10

# Change vocabulary size
MAX_FEATURES: int = 5000  # Change to 10000

# Change cache duration
CACHE_TTL: int = 3600  # 1 hour (in seconds)
```

---

## Testing Your Installation

```python
# Quick test in Python console
from data_loader import DataLoader
from recommendation_engine import RecommendationEngine

# Load data
game_data, similarity, image_data, game_index = DataLoader.load_all_data()

# Initialize engine
engine = RecommendationEngine(game_data, similarity, image_data, game_indx)

# Get recommendations
recs = engine.recommend("Stray", top_n=5)

# Print results
for rec in recs:
    print(f"{rec.game_name}: {rec.similarity_score:.2%}")
```

Expected output:
```
Similar Game 1: 95.3%
Similar Game 2: 87.2%
Similar Game 3: 82.1%
...
```

---

## API Usage (Programmatic)

```python
from recommendation_engine import RecommendationEngine
from data_loader import DataLoader

# Load data
data = DataLoader.load_all_data()

# Create engine
engine = RecommendationEngine(*data)

# Get recommendations
recommendations = engine.recommend("Game Name", top_n=10)

# Access results
for rec in recommendations:
    print(f"Game: {rec.game_name}")
    print(f"Score: {rec.similarity_score}")
    print(f"Image: {rec.image_url}")
    print(f"AppID: {rec.app_id}")
    print()
```

---

## Deployment

### HuggingFace Spaces
1. Create account at huggingface.co
2. Create new Space (Streamlit)
3. Upload all files
4. Done! Auto-deployed

### Docker
```bash
# Build
docker build -t steam-recommender .

# Run
docker run -p 8501:8501 steam-recommender

# Access at localhost:8501
```

---

## Performance Tips

1. **Use caching**: Already enabled by default
2. **Limit recommendations**: 6-12 optimal for UX
3. **Precompute similarity**: Done in preprocessing
4. **Use .npy format**: Already configured
5. **Monitor logs**: Check `logs/app.log`

---

## Getting Help

1. **Check logs**: `logs/app.log`
2. **Run tests**: `pytest test_recommendation.py -v`
3. **Verify installation**: `python setup.py`
4. **Read documentation**: `README.md`
5. **Check improvements**: `IMPROVEMENTS.md`

---

## Development Workflow

```bash
# 1. Make changes to code
vim recommendation_engine.py

# 2. Run tests
pytest test_recommendation.py -v

# 3. Test in app
streamlit run app.py

# 4. Commit changes
git add .
git commit -m "Your message"
git push
```

---

## Resource Requirements

### Minimum
- **RAM**: 4 GB
- **Storage**: 5 GB
- **CPU**: 2 cores
- **Python**: 3.8+

### Recommended
- **RAM**: 8 GB
- **Storage**: 10 GB
- **CPU**: 4 cores
- **Python**: 3.10+

---

## Dataset Stats (Example)

For the original Steam dataset:
- **Games**: 18,560
- **Date Range**: 2021-2023
- **Vocabulary**: 5,000 features
- **Matrix Size**: 1.3 GB
- **Processing Time**: ~15 minutes

---

## Support

- **Issues**: GitHub Issues
- **Email**: ashishprasad@gmail.com
- **LinkedIn**: [Ashish Prasad](https://www.linkedin.com/in/ashish-prasad-92223a228/)

---

**Happy recommending! 🎮✨**
