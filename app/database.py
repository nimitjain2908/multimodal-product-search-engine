import os
import pandas as pd
import chromadb
from PIL import Image
from app.config import AppConfig
from app.engine import MultiModalEngine
from tqdm import tqdm

class VectorDBManager:
    """Manages on-disk ChromaDB initialization, catalog ingestion, and querying."""
    
    def __init__(self):
        self.config = AppConfig()
        self.engine = MultiModalEngine()
        
        # Initialize the persistent ChromaDB client on your local hard drive
        self.chroma_client = chromadb.PersistentClient(path=str(self.config.CHROMA_DB_DIR))
        
        # Get or create our dedicated multi-modal collection room
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Explicitly instructs Chroma to use Cosine distance math
        )
        
    def ingest_catalog_subset(self, limit: int = 1000):
        """Iterates through data, extracts vectors in optimized batches, and saves to ChromaDB."""
        print(f"\n--- Reading Catalog Metadata Ledger ---")
        df = pd.read_csv(self.config.METADATA_CSV, on_bad_lines='skip')
        
        # Filter rows to verify the physical image actually exists on disk
        valid_rows = []
        for _, row in df.iterrows():
            img_name = f"{int(row['id'])}.jpg"
            if (self.config.IMAGES_DIR / img_name).exists():
                valid_rows.append(row)
                if len(valid_rows) >= limit: # Cap at our subset limit for safe prototyping
                    break
                    
        filtered_df = pd.DataFrame(valid_rows)
        print(f"Found {len(filtered_df)} valid records ready for database injection.")
        
        # Batch Ingestion configurations
        batch_size = self.config.BATCH_SIZE
        print(f"Starting database ingestion in chunks of {batch_size}...")
        
        for i in tqdm(range(0, len(filtered_df), batch_size)):
            chunk = filtered_df.iloc[i : i + batch_size]
            
            ids = []
            embeddings = []
            metadatas = []
            
            for _, row in chunk.iterrows():
                product_id = str(int(row['id']))
                img_path = self.config.IMAGES_DIR / f"{product_id}.jpg"
                
                try:
                    # Load raw image asset and pass it into our embedding engine class
                    with Image.open(img_path).convert("RGB") as img:
                        vector = self.engine.get_image_embedding(img)
                        
                    ids.append(product_id)
                    embeddings.append(vector)
                    
                    # Store descriptive metadata alongside the vector coordinate
                    metadatas.append({
                        "productName": str(row.get("productDisplayName", "Unknown")),
                        "category": str(row.get("masterCategory", "Unknown")),
                        "subCategory": str(row.get("subCategory", "Unknown")),
                        "gender": str(row.get("gender", "Unknown"))
                    })
                except Exception as e:
                    continue # Skip corrupted files safely
            
            # Upsert the processed batch directly into the persistent database collection
            if ids:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
        print(f"🎉 Successfully indexed and stored {self.collection.count()} records in ChromaDB!")

    def query_by_text(self, text_query: str, top_k: int = 3) -> dict:
        """Converts a string query into a vector and queries ChromaDB for the closest spatial matches."""
        # Calculate the text query vector coordinates
        query_vector = self.engine.get_text_embedding(text_query)
        
        # Execute the database vector space query
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        return results