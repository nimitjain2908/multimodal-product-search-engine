# Enterprise Multi-Modal Product Search Engine

A production-grade, high-performance cross-modal (Text-to-Image) retrieval system capable of executing semantic search vectors across a catalog of 44,000+ retail fashion assets. This architecture bypasses traditional keyword matching by projecting both natural language queries and product imagery into a shared high-dimensional vector space.

## 🛠️ Tech Stack & Core Dependencies
* **Core Language:** Python 3.11+
* **Deep Learning Framework:** PyTorch & Hugging Face Transformers (`CLIP-ViT-Base-Patch32`)
* **Vector Database:** ChromaDB (Configured for native Cosine Similarity matching)
* **Application Layer:** FastAPI (Asynchronous microservice router)
* **Web Gateway Gateway:** Uvicorn

## 🏗️ System Architecture & Data Flow

1. **Ingestion Pipeline:** Raw catalog images are loaded via Pillow, mapped through the CLIP visual transformer to extract 512-dimensional coordinates, geometrically L2-normalized, and upserted into a persistent local ChromaDB collection alongside structured metadata.
2. **Query Processing Layer:** Natural language text input strings are tokenized and processed via the CLIP text encoder to generate matching search coordinates.
3. **Vector Similarity Search:** ChromaDB executes an optimized HNSW index lookup using Cosine Distance space calculations to return the Top-K closest product matches.
4. **Media Streaming Delivery:** FastAPI utilizes asynchronous file operations (`aiofiles` and `FileResponse`) to stream physical high-resolution retail assets straight to the client browser via unified endpoints.

## 🚀 Production Engineering Highlights

### 1. Geometric L2 Vector Normalization
To guarantee flawless accuracy under Cosine Distance search configurations, all coordinate arrays are explicitly normalized to a unit length of 1.0 using L2 geometry routines before being indexed:

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$$

### 2. High-Performance Asynchronous I/O
The API layer leverages FastAPI's non-blocking `async` runtime alongside `aiofiles`. This keeps the microservice highly responsive, allowing it to handle concurrent user search requests without locking up main thread CPU cycles while waiting on disk reads.

## 📂 Project Structure
```text
GenAI RAG/
│
├── app/
│   ├── __init__.py
│   ├── config.py       # Centralized directory paths & hyperparameter configurations
│   ├── engine.py       # CLIP neural network inference and normalization class
│   ├── database.py     # ChromaDB collection handling and data ingestion loops
│   └── main.py         # Unified FastAPI application routing layer
│
├── Phase1_Exploration.ipynb  # Initial prototyping notebook
├── .gitignore          # Safeguards data files from repository commits
└── README.md           # Professional portfolio documentation