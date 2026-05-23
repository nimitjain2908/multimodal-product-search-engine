# 🛍️ Enterprise Multi-Modal Product Search Engine

An end-to-end AI-powered retail discovery platform utilizing a **CLIP Dual-Encoder Architecture** and a high-performance vector index to enable semantic, natural language searches over a catalog of **44,419 fashion assets**.

The project is engineered as a decoupled microservices architecture featuring a high-performance asynchronous retrieval API and an interactive, real-time visual web dashboard.

---

## 🏗️ System Architecture & Workflow

1. **Ingestion & Embedding Pipeline:** Processes 44k+ multi-modal catalog items. Text descriptions and images are projected into a unified 512-dimensional vector space using the pre-trained `CLIP-ViT-Base-Patch32` transformer model.
2. **L2 Normalization & Indexing:** All generated embeddings undergo rigorous $L2$ normalization to ensure mathematically precise cosine similarity calculations inside a local **ChromaDB** vector database.
3. **Asynchronous Serving Layer:** A highly efficient **FastAPI** backend exposes production endpoints for semantic query vectorization, top-$K$ vector proximity lookups, and direct binary image streaming from disk.
4. **Interactive Dashboard Layout:** A responsive **Streamlit** user interface captures human queries, communicates via REST protocols with the backend microservice, parses the nested metadata schema, and dynamically renders product assets in a clean grid layout.

---

## 🛠️ Tech Stack & Core Libraries

* **Deep Learning Framework:** PyTorch, Hug Face `transformers` (CLIP)
* **Vector Database:** ChromaDB (HNSW Graph Engine)
* **Backend Framework:** FastAPI, Uvicorn (Asynchronous I/O processing)
* **Frontend Dashboard:** Streamlit, Requests
* **DevOps Infrastructure:** Docker, Docker Compose (Multi-container orchestration blueprints included)

---

## 🚀 How to Run the System Natively

### Prerequisites
Ensure you have Python 3.10+ installed and your fashion dataset downloaded locally.

### 1. Initialize the Environment & Dependencies
Clone the repository, navigate to the project root, create your configuration files, and install the required packages:
```bash
pip install -r requirements.txt

2. Boot the FastAPI Backend Service
Launch the asynchronous API server using Uvicorn. This layer initializes the CLIP model, loads weights into memory, and mounts your local ChromaDB vector index directory:

Bash
uvicorn app.main:app --reload
The interactive API documentation panel will be accessible live at http://127.0.0.1:8000/docs.

3. Launch the Streamlit Frontend Interface
Open a separate terminal window or split your current terminal view, and execute the Streamlit engine to boot up your visual search dashboard:

Bash
streamlit run app_ui.py
The web dashboard will automatically open in your browser interface at http://localhost:8501.

🐋 Enterprise Containerization Blueprints
For production-grade cloud deployments (AWS, GCP, Azure), the project contains built-in Docker configuration files to eliminate environment mismatches entirely.

Dockerfile.backend: Packages the PyTorch, C++ build dependencies, and FastAPI execution stack into a lean, isolated Linux environment.

Dockerfile.frontend: Builds a lightweight container dedicated entirely to serving the Streamlit dashboard layout.

docker-compose.yml: Network-orchestrates both containers simultaneously over a secure virtual bridge network, mapping the local image cache cleanly via disk volumes.

To deploy the entire multi-container microservice ecosystem with a single command:

Bash
docker compose up --build