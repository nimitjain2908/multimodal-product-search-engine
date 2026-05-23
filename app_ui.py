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
                # 1. Make call to the FastAPI search endpoint with mapped parameter keys
                response = requests.get(f"{BACKEND_URL}/search", params={"q": query, "limit": top_k})
                
                if response.status_code == 200:
                    api_response = response.json()
                    
                    # Extract the list of result dictionaries from the top-level key
                    results = api_response.get("results", [])
                    
                    if not results:
                        st.info("No matching items found in the vector index.")
                    else:
                        st.success(f"Successfully retrieved top {len(results)} matches!")
                        
                        # Build a clean, responsive 4-column layout grid
                        cols = st.columns(4)
                        for idx, item in enumerate(results):
                            # Clean Extraction: Safely grab the product_id from the inner dictionary
                            prod_id = str(item.get("product_id", ""))
                            
                            # Grab details for beautiful display captions
                            details = item.get("details", {})
                            prod_name = details.get("product_name", "Unknown Product")
                            gender = details.get("gender_tag", "Unisex")
                            score = item.get("confidence_score", "0.0%")
                            
                            # Calculate exactly which column column this item belongs to
                            col_idx = idx % 4
                            with cols[col_idx]:
                                # Route directly to your live FastAPI binary stream endpoint
                                img_url = f"{BACKEND_URL}/product-image/{prod_id}"
                                
                                # Render the image cleanly using container width rules
                                st.image(img_url, use_container_width=True)
                                
                                # Render rich metadata cards underneath each asset
                                st.markdown(f"### ID: {prod_id}")
                                st.markdown(f"**Match:** `{score}`")
                                st.markdown(f"*{prod_name}*")
                                st.caption(f"Tag: {gender}")
                else:
                    st.error(f"Backend API returned an error status: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the Backend API at {BACKEND_URL}. Ensure your FastAPI server is online!")