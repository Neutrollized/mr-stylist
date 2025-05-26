# config.py
"""
Configuration settings for the Fashion Recommendation Application.

This module stores shared constants used across various parts of the application,
including AI model names, model parameters, prompts, thresholds, project settings,
and file paths.
"""

# Models
GEMINI_MODEL = "gemini-2.0-flash-001"  # Specifies the Gemini model version for multimodal tasks.
EMBEDDING_MODEL = "text-embedding-005"  # Specifies the text embedding model version.

# Model parameters for generative AI
TEMPERATURE = 0.1  # Controls randomness in generation. Lower is less random.
TOP_P = 0.8        # Nucleus sampling parameter.
TOP_K = 25         # Top-k sampling parameter.

# Prompts
STYLIST_PROMPT_EMBED = "Provide a few sentences describing the clothing's type, color, and style"
"""Prompt used for generating concise descriptions for embedding purposes."""

STYLIST_PROMPT_GENERAL = "Can you describe the clothes in the photo, including style, color, and any designs?  Make sure to only describe each individual article of clothing, and give a separate response."
"""General prompt used for describing clothing items in detail for recommendation or display."""

# Thresholds
COSINE_SCORE_THRESHOLD = 0.65  # Minimum cosine similarity score to consider a match.

# Google Cloud Project settings
PROJECT_ID = ""  # Target Google Cloud Project ID. Leave empty to allow auto-detection from gcloud or environment.
LOCATION = "us-central1"  # Default Google Cloud region for Vertex AI services.

# File paths and application settings
IMAGE_URI_PATH = "static/images/"     # Path to the directory containing wardrobe images.
WARDROBE_CSV_FILE = "mywardrobe.csv"  # Filename for the CSV storing wardrobe item metadata and embeddings.
UPLOAD_FOLDER = "temp/"               # Temporary folder for storing uploaded images for processing.
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'JPG', 'JPEG'}  # Set of allowed image file extensions for upload.
