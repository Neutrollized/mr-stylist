#! /usr/bin/env python3
"""
Embed Wardrobe Script.

This script processes images from a specified directory, generates textual descriptions
and embeddings for each image using Vertex AI models, and saves this metadata
into a CSV file. This CSV file serves as the "wardrobe database" for the
recommendation system.

Key functionalities:
- Initializes Vertex AI services.
- Scans a directory for images.
- For each image:
    - Resizes the image.
    - Generates a textual description using a multimodal AI model.
    - Generates a numerical embedding for the description using a text embedding model.
- Saves the collected data (image URI, description, embedding) into a CSV file.
- Includes a delay between processing images to respect API rate limits.
"""
import os
# import sys # sys is no longer used directly in this script
import logging # For logging status and errors
import glob
import time
from typing import Dict, List, Union # For type hinting

import pandas as pd
import numpy as np # For potential use if embeddings were returned as np.ndarray and needed specific handling

from config import (
    STYLIST_PROMPT_EMBED, # Prompt specific for generating embeddings
    IMAGE_URI_PATH,        # Path to wardrobe images
    WARDROBE_CSV_FILE,     # Output CSV filename
    # AI model names (GEMINI_MODEL, EMBEDDING_MODEL) and parameters (TEMPERATURE, etc.)
    # are used by vertex_ai_utils.py, so not directly imported here.
    # PROJECT_ID and LOCATION are also handled by vertex_ai_utils.py.
)

from helpers.image_utils import image_resize
from vertex_ai_utils import (
    init_vertex_ai,
    generate_text,
    get_text_embedding_from_text_embedding_model,
    # No direct need for get_multimodal_model or get_generation_config here,
    # as they are encapsulated within generate_text.
)

#-----------------------------------
# Initialize Vertex AI (called once at script start)
#-----------------------------------
init_vertex_ai()
# logging.info(f"Vertex AI Initialized with Project ID: {vertex_ai_utils.get_project_id()}") # Example

#-----------------------------------------
# Function to Process a Single Image
#-----------------------------------------
def process_image_for_wardrobe(image_file_path: str, stylist_prompt: str) -> Dict[str, Union[str, List[float]]]:
    """
    Processes a single image: resizes, generates description, and creates embedding.

    Args:
        image_file_path: The absolute or relative path to the image file.
        stylist_prompt: The prompt to use for generating the image description.

    Returns:
        A dictionary containing the processed image data:
        {
            'image_uri': str,  // The path to the (potentially resized) image.
            'image_description_text': str, // The generated textual description.
            'image_description_text_embedding': List[float] // The numerical embedding of the description.
        }
        Returns an empty dictionary if a critical error occurs during processing.
        
    Raises:
        FileNotFoundError: If the image_file_path does not exist (handled by image_resize).
        Exception: Propagates exceptions from AI model calls or image processing if not handled within.
    """
    logging.info(f"Processing image: {image_file_path}")
    
    try:
        # Resize the image (modifies in place and returns path)
        resized_image_file_path = image_resize(image_file_path, 1024) 
        
        # Generate text description using the shared multimodal model and generation config
        description_text = generate_text(
            image_uri=resized_image_file_path,
            prompt=stylist_prompt,
        )
        
        # Generate text embedding for the description
        # This function returns Union[List[float], np.ndarray]. We expect List[float] by default.
        embedding_result = get_text_embedding_from_text_embedding_model(
            text=description_text,
            return_array=False # Explicitly ensuring List[float]
        )
        # Ensure it's a list, as np.ndarray is not directly JSON serializable for Pandas to_csv as easily
        if isinstance(embedding_result, np.ndarray):
            image_description_text_embedding: List[float] = embedding_result.tolist()
        else:
            image_description_text_embedding: List[float] = embedding_result

        return {
            'image_uri': resized_image_file_path,
            'image_description_text': description_text,
            'image_description_text_embedding': image_description_text_embedding,
        }
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found error for {image_file_path}: {fnf_error}")
        raise # Re-raise to be caught by the main loop's error handler
    except Exception as e:
        logging.error(f"Failed to process image {image_file_path}: {e}")
        raise # Re-raise to be caught by the main loop's error handler

#-----------------------------------------
# Main Script Execution
#-----------------------------------------
def main() -> None:
    """
    Main function to find images in the specified directory, process each image
    to generate descriptions and embeddings, and save this metadata to a CSV file.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting wardrobe embedding process...")
    
    wardrobe_data_list: List[Dict[str, Union[str, List[float]]]] = []
    
    # Define glob patterns for common image extensions.
    # Using os.path.join for platform-independent path construction.
    # Configurable via ALLOWED_EXTENSIONS from config.py if needed in future.
    image_extensions = ["*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    image_files_to_process: List[str] = []
    for ext in image_extensions:
        pattern = os.path.join(IMAGE_URI_PATH, ext)
        image_files_to_process.extend(glob.glob(pattern))

    if not image_files_to_process:
        logging.warning(f"No images found in '{IMAGE_URI_PATH}' with patterns {image_extensions}. Exiting.")
        return

    logging.info(f"Found {len(image_files_to_process)} images to process.")

    for image_path in image_files_to_process:
        try:
            processed_data = process_image_for_wardrobe(image_path, STYLIST_PROMPT_EMBED)
            if processed_data: # Ensure data was actually processed
                wardrobe_data_list.append(processed_data)
                logging.info(f"Successfully processed: {image_path}")
        except Exception as e:
            # Error is already logged in process_image_for_wardrobe,
            # but we can add context here or decide to skip the image.
            logging.error(f"Skipping image {image_path} due to error during its processing: {e}")
        
        # API call delay - consider making this configurable
        # For a small number of images, this might be very slow.
        # For larger batches, this helps manage rate limits.
        logging.info("Waiting for 8 seconds before next API call...")
        time.sleep(8) 
        
    if not wardrobe_data_list:
        logging.warning("No data was successfully processed. Output CSV will be empty or not created.")
        return

    # Create Pandas DataFrame from the list of dictionaries
    image_metadata_df = pd.DataFrame(wardrobe_data_list)
    
    # Save DataFrame to CSV
    try:
        image_metadata_df.to_csv(WARDROBE_CSV_FILE, index=False)
        logging.info(f"Wardrobe metadata saved to {WARDROBE_CSV_FILE}")
        # logging.info(image_metadata_df.head()) # Print head for quick check
    except Exception as e:
        logging.error(f"Error saving DataFrame to CSV '{WARDROBE_CSV_FILE}': {e}")

if __name__ == "__main__":
    main()
