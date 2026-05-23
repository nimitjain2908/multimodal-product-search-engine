from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from app.database import VectorDBManager

# 1. Instantiate the Global Web Application Layer
app = FastAPI(
    title="Enterprise Multi-Modal Product Search Engine",
    description="Production-grade FastAPI service querying high-dimensional catalog vectors using ChromaDB and CLIP.",
    version="1.0.0"
)

# 2. Initialize our Singleton Vector Database Manager 
# This spins up the index and loads the model into RAM right when the web server boots up
print("\n🔥 Bootstrapping core neural assets and initializing database registry...")
db_manager = VectorDBManager()

@app.get("/")
def read_root():
    """Health check endpoint to ensure our microservice is running normally."""
    return {
        "status": "online",
        "database_record_count": db_manager.collection.count(),
        "engine_specification": db_manager.config.MODEL_NAME
    }

@app.get("/search")
async def search_catalog(
    q: str = Query(..., description="The natural language text query description of the product you are seeking"),
    limit: int = Query(3, description="The maximum number of top-k target matches to pull from the database collection")
):
    """
    Asynchronous Production API endpoint handling cross-modal text-to-image semantic query processing.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query text string parameter cannot be empty.")
        
    try:
        # Run our verified vector space lookup matrix computation
        search_results = db_manager.query_by_text(text_query=q, top_k=limit)
        
        # Unpack the response payload safely
        ids = search_results['ids'][0]
        distances = search_results['distances'][0]
        metadatas = search_results['metadatas'][0]
        
        formatted_response = []
        for i in range(len(ids)):
            # Convert raw distance array results to human confidence ratings
            confidence_score = (1.0 - distances[i]) * 100
            
            formatted_response.append({
                "rank": i + 1,
                "product_id": ids[i],
                "confidence_score": f"{confidence_score:.2f}%",
                "details": {
                    "product_name": metadatas[i].get("productName"),
                    "master_category": metadatas[i].get("category"),
                    "sub_category": metadatas[i].get("subCategory"),
                    "gender_tag": metadatas[i].get("gender")
                }
            })
            
        return {
            "query": q,
            "total_matches_returned": len(formatted_response),
            "results": formatted_response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Vector Database Engine Error: {str(e)}")

@app.get("/product-image/{product_id}")
def get_product_image(product_id: str):
    """
    Looks up a specific Product ID and streams the actual visual 
    image file from the local hard drive directory directly to the user.
    """
    # 1. Build the absolute path to where the image lives on your laptop
    image_file_path = db_manager.config.IMAGES_DIR / f"{product_id}.jpg"
    
    # 2. Check if the file physically exists on your disk
    if not image_file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image for Product ID {product_id} not found on disk.")
        
    # 3. Stream the raw image asset file straight to the browser screen
    return FileResponse(image_file_path, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    print("\n--- Running Bulk Ingestion Pipeline ---")
    # This lifts the safety cap and triggers the 44,000+ full catalog index
    db_manager.ingest_catalog_subset(limit=45000)