import streamlit as st
import requests
import os

# Set up page configurations
st.set_page_config(page_title="Multi-Modal Semantic Search", layout="wide")

st.title("🛍️ Enterprise Multi-Modal Product Search Engine")
st.write("Powered by CLIP-ViT-Base-Patch32 Dual Encoder & ChromaDB Vector Index")

# API Endpoint definition (Will point to our backend service)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Search Input Layout
query = st.text_input("Enter your semantic search phrase:", placeholder="e.g., casual blue check shirt for summer weekends")
top_k = st.slider("Number of results to fetch (Top-K):", min_value=1, max_value=12, value=4)

if st.button("Search Catalog", type="primary"):
    if query.strip() == "":
        st.warning("Please enter a valid search term.")
    else:
        with st.spinner("Vectorizing query and exploring high-dimensional embedding space..."):
            try:
                # 1. Make call to the FastAPI search endpoint
                response = requests.post(f"{BACKEND_URL}/search", json={"query": query, "top_k": top_k})
                
                if response.status_code == 200:
                    results = response.json() # Expecting a list of item dictionaries
                    
                    if not results:
                        st.info("No matching items found in the vector index.")
                    else:
                        st.success(f"Successfully retrieved top {len(results)} matches!")
                        
                        # 2. Build a responsive multi-column grid layout for images
                        cols = st.columns(4)
                        for idx, item in enumerate(results):
                            prod_id = item.get("product_id")
                            prod_name = item.get("product_name", "Unknown Product")
                            distance = item.get("distance", 0.0)
                            
                            # Calculate column assignment
                            col_idx = idx % 4
                            with cols[col_idx]:
                                # Call our live binary image stream endpoint
                                img_url = f"{BACKEND_URL}/product-image/{prod_id}"
                                st.image(img_url, use_column_width=True)
                                st.caption(f"**ID:** {prod_id}\n\n**Dist:** {distance:.4f}")
                else:
                    st.error(f"Backend API returned an error status: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the Backend API at {BACKEND_URL}. Ensure your FastAPI server is online!")