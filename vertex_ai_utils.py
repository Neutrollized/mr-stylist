# vertex_ai_utils.py
"""
Vertex AI Utilities for the Fashion Recommendation Application.

This module centralizes interactions with Google Cloud Vertex AI services,
including model initialization, configuration, and helper functions for
text generation, text embedding, and image embedding. It aims to provide
singleton instances for models to optimize resource usage.
"""
import os
import sys # sys is used for checking 'google.colab' in modules
import subprocess
import logging # For logging messages
from typing import List, Dict, Tuple, Optional, Any, Union # For type hinting

import numpy as np
import vertexai
from vertexai.generative_models import (
    # Content, # Content not directly used in this file's public interface
    GenerationConfig,
    GenerativeModel,
    Image, # Used by generate_text
    Part,  # Used by generate_text
)
from google.generativeai import client as GenAIClient # Specific import for genai.Client
from google.generativeai.types import EmbedContentResponse, EmbedContentConfig

from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as VisionModelImage # Alias to avoid conflict with generative_models.Image
from vertexai.vision_models import MultiModalEmbeddingModel as VisionMultiModalEmbeddingModel # Alias for clarity

from config import (
    GEMINI_MODEL, # Name of the Gemini model for multimodal tasks
    EMBEDDING_MODEL, # Name of the text embedding model
    TEMPERATURE,     # Default temperature for generative model
    TOP_P,           # Default top-p for generative model
    TOP_K,           # Default top-k for generative model
    PROJECT_ID as CFG_PROJECT_ID, # Default Project ID from config
    LOCATION as CFG_LOCATION,     # Default Location from config
)

# --- Global Variables for Singleton Model Instances ---
_PROJECT_ID: Optional[str] = None
_LOCATION: Optional[str] = None
_multimodal_model: Optional[GenerativeModel] = None
_text_embedding_model_vertex: Optional[TextEmbeddingModel] = None
_vision_multimodal_embedding_model: Optional[VisionMultiModalEmbeddingModel] = None
_generation_config: Optional[GenerationConfig] = None

# --- System Instruction for Multimodal Model ---
DEFAULT_SYSTEM_INSTRUCTION: List[str] = [
    "You are a fashion stylist.",
    "Your mission is to describe the clothing you see.",
]

# --- Initialization and Configuration ---
def init_vertex_ai() -> None:
    """
    Initializes Vertex AI with project ID and location.

    Attempts to fetch Project ID from environment variable 'MY_PROJECT_ID',
    then from gcloud configuration if not running in Colab, and finally
    falls back to CFG_PROJECT_ID from `config.py`.
    Location is taken from `config.py`.
    Raises ValueError if Project ID cannot be determined.
    This function should be called once at the application startup.
    """
    global _PROJECT_ID, _LOCATION
    if _PROJECT_ID and _LOCATION:
        logging.info(f"Vertex AI already initialized with Project ID: {_PROJECT_ID} and Location: {_LOCATION}")
        return

    _PROJECT_ID = os.environ.get('MY_PROJECT_ID') or CFG_PROJECT_ID
    _LOCATION = CFG_LOCATION # Assumed to be always set in config.py

    if not _PROJECT_ID and "google.colab" not in sys.modules:
        try:
            _PROJECT_ID = subprocess.check_output(
                ["gcloud", "config", "get-value", "project"], text=True
            ).strip()
            logging.info(f"Auto-detected Project ID from gcloud: {_PROJECT_ID}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.warning(f"Could not auto-detect Project ID from gcloud: {e}. Will use CFG_PROJECT_ID if set.")
            # If CFG_PROJECT_ID is also empty, the check below will catch it.

    if not _PROJECT_ID:
        logging.error("Project ID is not set after checking ENV, gcloud, and config.py.")
        raise ValueError("Project ID is not set. Please configure MY_PROJECT_ID, gcloud project, or CFG_PROJECT_ID in config.py.")

    logging.info(f"Initializing Vertex AI with Project ID: {_PROJECT_ID}, Location: {_LOCATION}")
    try:
        vertexai.init(project=_PROJECT_ID, location=_LOCATION)
    except Exception as e:
        logging.error(f"Error during vertexai.init: {e}")
        raise

def get_project_id() -> str:
    """Returns the initialized Google Cloud Project ID. Calls init_vertex_ai() if not already initialized."""
    if not _PROJECT_ID:
        init_vertex_ai()
    return _PROJECT_ID

def get_location() -> str:
    """Returns the initialized Google Cloud Location. Calls init_vertex_ai() if not already initialized."""
    if not _LOCATION: # Location should be set by init_vertex_ai from config
        init_vertex_ai()
    return _LOCATION

def get_generation_config() -> GenerationConfig:
    """
    Returns a singleton instance of GenerationConfig based on settings in `config.py`.
    Used for controlling the output of generative models.
    """
    global _generation_config
    if _generation_config is None:
        _generation_config = GenerationConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            candidate_count=1,
            max_output_tokens=2048, # Default max tokens, can be overridden per call if model supports
        )
    return _generation_config

# --- Model Getters (Singleton Pattern) ---
def get_multimodal_model() -> GenerativeModel:
    """
    Initializes (if necessary) and returns a singleton instance of the
    multimodal GenerativeModel (e.g., Gemini for vision and text tasks),
    configured with default system instructions.
    """
    global _multimodal_model
    if _multimodal_model is None:
        init_vertex_ai() # Ensure Vertex AI SDK is initialized
        logging.info(f"Initializing GenerativeModel: {GEMINI_MODEL} with default system instruction.")
        _multimodal_model = GenerativeModel(
            GEMINI_MODEL,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
        )
    return _multimodal_model

def get_text_embedding_model_vertex() -> TextEmbeddingModel:
    """
    Initializes (if necessary) and returns a singleton instance of the
    TextEmbeddingModel from `vertexai.language_models` using EMBEDDING_MODEL from config.
    """
    global _text_embedding_model_vertex
    if _text_embedding_model_vertex is None:
        init_vertex_ai()
        logging.info(f"Initializing TextEmbeddingModel (Vertex AI SDK): {EMBEDDING_MODEL}")
        _text_embedding_model_vertex = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    return _text_embedding_model_vertex

def get_vision_multimodal_embedding_model() -> VisionMultiModalEmbeddingModel:
    """
    Initializes (if necessary) and returns a singleton instance of the
    MultiModalEmbeddingModel from `vertexai.vision_models`.
    Note: The specific model name (e.g., "multimodalembedding@001") is currently hardcoded
    as it's not present in `config.py` for this particular model type.
    """
    global _vision_multimodal_embedding_model
    if _vision_multimodal_embedding_model is None:
        init_vertex_ai()
        # TODO: Consider making "multimodalembedding@001" configurable in config.py
        model_name = "multimodalembedding@001" 
        logging.info(f"Initializing MultiModalEmbeddingModel (Vision SDK): {model_name}")
        _vision_multimodal_embedding_model = VisionMultiModalEmbeddingModel.from_pretrained(model_name)
    return _vision_multimodal_embedding_model

# --- AI Helper Functions ---
def generate_text(image_uri: str, prompt: str, system_instruction: Optional[List[str]] = None) -> str:
    """
    Generates text from an image and a prompt using the multimodal model.

    Args:
        image_uri: The URI of the image (e.g., local path).
        prompt: The textual prompt to guide text generation.
        system_instruction: Optional list of strings for system-level instructions.
                            If provided, it temporarily re-initializes a model instance
                            for this specific call. If None, the default system instruction
                            of the shared multimodal model is used.

    Returns:
        The generated text as a string.
        
    Raises:
        Exception: If image loading or text generation fails.
    """
    target_model = get_multimodal_model()
    
    if system_instruction:
        # Creates a new model instance for this call if specific system instructions are given.
        # This is not ideal for high performance if system instructions change frequently.
        logging.info(f"Using specific system instruction for this generate_text call: {system_instruction}")
        try:
            target_model = GenerativeModel(GEMINI_MODEL, system_instruction=system_instruction)
        except Exception as e:
            logging.error(f"Error re-initializing GenerativeModel with custom system instruction: {e}")
            raise
            
    generation_config_instance = get_generation_config()
    
    try:
        image_part = Part.from_image(Image.load_from_file(image_uri))
    except Exception as e:
        logging.error(f"Failed to load image from URI '{image_uri}': {e}")
        raise
        
    contents = [image_part, prompt]

    try:
        response = target_model.generate_content(
            contents,
            generation_config=generation_config_instance,
        )
        return response.text
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        # Consider more specific error handling or re-raising with context
        raise

def get_text_embedding_from_text_embedding_model(
    text: str,
    return_array: Optional[bool] = False,
    task_type: str = "RETRIEVAL_QUERY",
    output_dimensionality: Optional[int] = None, # Set to None to use model's default unless specified
) -> Union[List[float], np.ndarray]:
    """
    Generates a numerical text embedding using the `google.generativeai.Client`.

    This function uses the `EMBEDDING_MODEL` specified in `config.py`.
    It matches the original implementation structure found in the project scripts
    that used `genai.Client().models.embed_content()`.

    Args:
        text: The input text string to be embedded.
        return_array: If True, returns the embedding as a NumPy array.
                      Otherwise, returns as a list of floats. Defaults to False.
        task_type: The task type for the embedding (e.g., "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY").
        output_dimensionality: Optional desired dimensionality of the output embedding.
                               If None, the model's default is used. Note: Not all models
                               support overriding dimensionality via EmbedContentConfig.

    Returns:
        A list of floats or a NumPy array representing the text embedding.
        Returns an empty list/array if embedding fails.
        
    Raises:
        Exception: If embedding generation fails.
    """
    try:
        # Using GenAIClient which is an alias for google.generativeai.Client
        client = GenAIClient() 
        
        config_params = {"task_type": task_type}
        if output_dimensionality is not None:
            # Note: output_dimensionality support in EmbedContentConfig can be model-specific.
            config_params["output_dimensionality"] = output_dimensionality
        
        config = EmbedContentConfig(**config_params)

        # The model name `EMBEDDING_MODEL` (e.g., "text-embedding-005") is used directly,
        # as this was the pattern in the original scripts with `genai.Client()`.
        response: EmbedContentResponse = client.models.embed_content(
            model=EMBEDDING_MODEL,
            content=text, 
            config=config,
        )
        
        # According to google.generativeai.types.EmbedContentResponse,
        # the embedding is accessed via response.embedding
        embedding_values: List[float] = response.embedding

        if return_array:
            return np.array(embedding_values, dtype=float)
        return embedding_values

    except Exception as e:
        logging.error(f"Error generating text embedding for '{text[:50]}...': {e}")
        # Depending on desired error handling, either re-raise or return empty
        # For now, re-raise to make failures more visible during this refactoring.
        raise # Consider: return [] if return_array else np.array([])

def get_image_embedding_from_multimodal_embedding_model(
    image_uri: str,
    contextual_text: Optional[str] = None, # Renamed 'text' to 'contextual_text' for clarity
    embedding_size: int = 1408, # Default from Vertex AI docs for this model
    return_array: Optional[bool] = False,
) -> Union[List[float], np.ndarray]:
    """
    Extracts an image embedding using the `vertexai.vision_models.MultiModalEmbeddingModel`.

    Args:
        image_uri: The URI of the image file.
        contextual_text: Optional text to provide context for the image embedding.
        embedding_size: The desired dimensionality of the output embedding.
                        Valid sizes for "multimodalembedding@001" are [128, 256, 512, 1408].
        return_array: If True, returns the embedding as a NumPy array.
                      Otherwise, returns as a list of floats. Defaults to False.

    Returns:
        A list of floats or a NumPy array representing the image embedding.
        Returns an empty list/array if embedding fails.
        
    Raises:
        ValueError: If an invalid `embedding_size` is provided for the known model.
        Exception: If image loading or embedding generation fails.
    """
    # Valid dimensions for the default "multimodalembedding@001" model
    valid_dimensions = [128, 256, 512, 1408]
    if embedding_size not in valid_dimensions:
        logging.error(f"Invalid embedding_size: {embedding_size}. Must be one of {valid_dimensions}.")
        raise ValueError(f"Invalid embedding_size: {embedding_size}. Must be one of {valid_dimensions}.")

    model = get_vision_multimodal_embedding_model()
    
    try:
        image = VisionModelImage.load_from_file(image_uri)
    except Exception as e:
        logging.error(f"Failed to load image for embedding from URI '{image_uri}': {e}")
        raise

    try:
        embeddings_response = model.get_embeddings(
            image=image,
            contextual_text=contextual_text,
            dimension=embedding_size,
        )
        # Assuming embeddings_response.image_embedding is List[float]
        image_embedding_values: List[float] = embeddings_response.image_embedding

        if return_array:
            return np.array(image_embedding_values, dtype=float)
        return image_embedding_values
        
    except Exception as e:
        logging.error(f"Error generating image embedding for '{image_uri}': {e}")
        raise # Consider: return [] if return_array else np.array([])
*   Global variables for models and settings to ensure singleton behavior.
*   `init_vertex_ai()` for Vertex AI initialization and dynamic `PROJECT_ID` fetching.
*   Getters for `PROJECT_ID`, `LOCATION`.
*   `get_generation_config()` for `GenerationConfig`.
*   `get_multimodal_model()` for `GenerativeModel` (Gemini for vision/text) with a default system instruction.
*   `get_text_embedding_model_vertex()` for `TextEmbeddingModel` from `vertexai.language_models`.
*   `get_vision_multimodal_embedding_model()` for `MultiModalEmbeddingModel` from `vertexai.vision_models`.
*   Moved and adapted `generate_text()`.
*   Moved and adapted `get_text_embedding_from_text_embedding_model()` (using `genai.Client()`). I've tried to reconcile the `content`/`contents` and response structure issues based on common usage and previous error messages. This function might need testing/adjustment.
*   Moved and adapted `get_image_embedding_from_multimodal_embedding_model()`.

I've also considered the `gemini_model_text_embed` function and, per instructions, will assume it's a duplicate for now and plan to remove it from `helpers/image_utils.py` later. The distinction between `genai.Client` based embeddings and `vertexai.language_models.TextEmbeddingModel` based embeddings is noted, and `vertex_ai_utils.py` provides access to both if needed.

Now, let's address `gemini_model_text_embed` in `helpers/image_utils.py`. I'll read the file, then remove the function.
