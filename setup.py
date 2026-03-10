"""
Setup script for Steam Game Recommendation System
Run this after installing requirements.txt
"""
import os
from pathlib import Path
import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs']
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        
        # Create .gitkeep to preserve empty directories
        gitkeep = path / '.gitkeep'
        gitkeep.touch()
        
        logger.info(f"Created directory: {directory}")


def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")
    
    try:
        # Download required packages
        packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
        
        for package in packages:
            try:
                nltk.download(package, quiet=True)
                logger.info(f"Downloaded NLTK package: {package}")
            except Exception as e:
                logger.warning(f"Could not download {package}: {e}")
        
        logger.info("NLTK data download complete")
        
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")


def verify_installation():
    """Verify that all required packages are installed"""
    logger.info("Verifying installation...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'nltk',
        'scipy',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} - NOT FOUND")
    
    if missing_packages:
        logger.error(f"\nMissing packages: {', '.join(missing_packages)}")
        logger.error("Please install: pip install -r requirements.txt")
        return False
    
    logger.info("\n✓ All required packages are installed!")
    return True


def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("\n1. Prepare your data:")
    print("   - Get Steam games CSV with required columns")
    print("   - Place it in the project directory")
    print("\n2. Run preprocessing:")
    print("   python preprocessing.py")
    print("\n3. Launch the app:")
    print("   streamlit run app.py")
    print("\n4. Run tests:")
    print("   pytest test_recommendation.py -v")
    print("\n" + "="*70)


def main():
    """Main setup function"""
    print("="*70)
    print("Steam Game Recommendation System - Setup")
    print("="*70 + "\n")
    
    # Create directory structure
    create_directory_structure()
    print()
    
    # Download NLTK data
    download_nltk_data()
    print()
    
    # Verify installation
    if verify_installation():
        print_next_steps()
    else:
        print("\n❌ Setup incomplete. Please install missing packages.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
