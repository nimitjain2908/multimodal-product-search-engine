import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from app.config import AppConfig

class MultiModalEngine:
    """Handles production-grade vector extraction for both images and text queries."""
    
    def __init__(self):
        # Load our shared configurations
        self.config = AppConfig()
        
        print(f"Initializing Multi-Modal Model: {self.config.MODEL_NAME}...")
        # Load the core processor and model weights from Hugging Face
        self.processor = CLIPProcessor.from_pretrained(self.config.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.config.MODEL_NAME)
        
        # Explicitly set model to evaluation mode for stable inference
        self.model.eval()
        
    def get_image_embedding(self, pil_image: Image.Image) -> list:
        """Transforms a raw PIL Image into an L2-Normalized 512-dimensional list."""
        # 1. Transform raw image into standardized tensor matrix
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # 2. Extract vector coordinates safely through weights
        with torch.no_grad():
            wrapped_output = self.model.get_image_features(**inputs)
            
            # Handle object wrapper safely
            if isinstance(wrapped_output, torch.Tensor):
                raw_tensor = wrapped_output
            else:
                raw_tensor = getattr(wrapped_output, "pooler_output", wrapped_output[0])
            
            # Escape the 3D patch grid trap if it triggers
            if len(raw_tensor.shape) == 3:
                raw_tensor = self.model.visual_projection(raw_tensor[:, 0, :])
                
            # Perform our geometric L2 Normalization math to squash length to 1.0
            normalized_tensor = raw_tensor / raw_tensor.norm(p=2, dim=-1, keepdim=True)
            
        # Convert the PyTorch tensor down to a standard Python list for database storage
        return normalized_tensor.squeeze(0).tolist()

    def get_text_embedding(self, text_query: str) -> list:
        """Transforms a natural language search query string into an L2-Normalized 512-dimensional list."""
        # 1. Tokenize query text string
        inputs = self.processor(text=[text_query], padding=True, return_tensors="pt")
        
        # 2. Map tokens through Text Encoder layers
        with torch.no_grad():
            wrapped_text_output = self.model.get_text_features(**inputs)
            
            # Handle object wrapper safely
            if isinstance(wrapped_text_output, torch.Tensor):
                raw_text_tensor = wrapped_text_output
            else:
                raw_text_tensor = getattr(wrapped_text_output, "pooler_output", wrapped_text_output[0])
                
            # Perform geometric L2 Normalization
            normalized_text_tensor = raw_text_tensor / raw_text_tensor.norm(p=2, dim=-1, keepdim=True)
            
        return normalized_text_tensor.squeeze(0).tolist()