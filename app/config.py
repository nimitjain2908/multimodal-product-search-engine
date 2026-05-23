import os
from pathlib import Path

class AppConfig:
    """Manages system paths, model configurations, and production constants."""
    
    # 1. Base Directory of the Project (Points to your GenAI RAG folder)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 2. Local Kaggle Dataset Paths (Targeting your specific machine paths)
    KAGGLE_DATA_DIR = Path(r"C:\Users\hp\.cache\kagglehub\datasets\paramaggarwal\fashion-product-images-small\versions\1")
    IMAGES_DIR = KAGGLE_DATA_DIR / "images"
    METADATA_CSV = KAGGLE_DATA_DIR / "styles.csv"
    
    # 3. Vector Database Storage Config
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    COLLECTION_NAME = "warehouse_catalog"
    
    # 4. Multi-Modal Model Specifications
    MODEL_NAME = "openai/clip-vit-base-patch32"
    VECTOR_DIMENSION = 512
    BATCH_SIZE = 64