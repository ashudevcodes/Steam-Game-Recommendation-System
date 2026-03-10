# Steam Game Recommendation System - Dockerfile
# Multi-stage build for optimized image size

FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('wordnet', quiet=True)"

# Final stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/nltk_data /root/nltk_data

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY config.py .
COPY data_loader.py .
COPY recommendation_engine.py .
COPY app.py .

# Create directories
RUN mkdir -p data models logs

# Copy pre-processed data files (if available)
# Uncomment these if you have pre-processed data
# COPY data/game_data.pkl data/
# COPY data/similarity.npy data/
# COPY data/image_data.csv data/
# COPY models/vectorizer.pkl models/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 
