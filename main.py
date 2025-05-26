#! /usr/bin/env python3
"""
Flask Web Application for Fashion Outfit Recommendation.

This application allows users to upload an image of a clothing item. It then
processes this image to understand its characteristics and recommends similar
or complementary items from a pre-defined wardrobe.

The main functionalities include:
- Serving a simple web interface for image uploads.
- Handling file uploads and temporary storage.
- Orchestrating the recommendation logic using helper and utility modules:
    - Generating descriptions for the uploaded image.
    - Filtering descriptions for relevance.
    - Querying a wardrobe database for similar items based on embeddings.
- Displaying recommended item images to the user.
- Centralized Vertex AI initialization and error handling.
"""
import os
import logging
import time # For retry delay
# from tempfile import NamedTemporaryFile # Not used as direct save/remove is implemented
from typing import List, Dict, Any, Union

import pandas as pd
from flask import Flask, render_template, request, Response # Added Response for return type hint
from werkzeug.datastructures import FileStorage # For type hinting uploaded_file_obj

# Config imports
from config import (
    STYLIST_PROMPT_GENERAL,    # Prompt for generating descriptions of uploaded images
    WARDROBE_CSV_FILE,         # Path to the wardrobe data CSV
    UPLOAD_FOLDER as CFG_UPLOAD_FOLDER,        # Folder for temporary uploads
    ALLOWED_EXTENSIONS as CFG_ALLOWED_EXTENSIONS, # Allowed image extensions
)

# Helper imports
# image_resize is used by get_reference_image_description in recommender.py, so not directly here.
from helpers.clothing_utils import CLOTHING_CATEGORIES # For filtering descriptions

# Recommender function imports (these now encapsulate more logic)
from recommender import (
    get_reference_image_description, # Generates descriptions for an image
    filter_image_descriptions,       # Filters those descriptions
    get_similar_text_from_query,     # Finds similar items from wardrobe
)

# Vertex AI utilities
from vertex_ai_utils import init_vertex_ai # Only init_vertex_ai is directly needed in main.py

#-----------------------------------
# Initialize Vertex AI (call once at application startup)
#-----------------------------------
init_vertex_ai()

#-----------------------------------------
# Flask App Setup & Configuration
#-----------------------------------------
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = CFG_UPLOAD_FOLDER  # Flask config for upload folder
app.config['ALLOWED_EXTENSIONS'] = CFG_ALLOWED_EXTENSIONS

#-----------------------------------------
# File Handling Helper Functions
#-----------------------------------------
def allowed_file(filename: str) -> bool:
  """
  Checks if the uploaded file has an allowed extension.

  Args:
      filename: The name of the uploaded file.

  Returns:
      True if the file extension is allowed, False otherwise.
  """
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_image(uploaded_file_obj: FileStorage) -> str:
    """
    Saves the uploaded file to a temporary path within the UPLOAD_FOLDER.
    
    The filename is preserved. In a high-concurrency environment, using UUIDs or
    other methods for unique filenames would be more robust.

    Args:
        uploaded_file_obj: The FileStorage object from Flask's request.files.

    Returns:
        The full path to where the file was saved.
        
    Raises:
        OSError: If directory creation or file saving fails.
    """
    upload_dir = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    
    # Using original filename; consider unique names for production
    filepath = os.path.join(upload_dir, uploaded_file_obj.filename)
    uploaded_file_obj.save(filepath)
    return filepath

#-----------------------------------------
# Recommendation Service Function
#-----------------------------------------
def get_recommendations_for_image(image_path: str) -> List[str]:
    """
    Orchestrates the logic to get outfit recommendations for a given image path.

    This involves loading wardrobe data, generating and filtering descriptions for
    the input image, querying for similar items, and formatting results.

    Args:
        image_path: The path to the (temporarily saved) uploaded image.

    Returns:
        A list of image URIs (strings) for the recommended items.
        These URIs are adjusted to be relative to the 'static/' folder for web display.
        Returns an empty list if no recommendations are found or if a critical error occurs.
        
    Raises:
        FileNotFoundError: If the WARDROBE_CSV_FILE is not found.
        Exception: Propagates other exceptions from underlying functions (e.g., AI model errors).
    """
    recommended_image_uris: List[str] = []
    
    try:
        wardrobe_df = pd.read_csv(
            WARDROBE_CSV_FILE,
            converters={"image_description_text_embedding": lambda x: [float(i) for i in x.strip("[]").split(",")]}
        )
        if wardrobe_df.empty:
            logging.warning(f"Wardrobe data file '{WARDROBE_CSV_FILE}' is empty. Cannot provide recommendations.")
            return [] 
    except FileNotFoundError:
        logging.error(f"CRITICAL: Wardrobe data file '{WARDROBE_CSV_FILE}' not found. Ensure embed_wardrobe.py has run.")
        raise # Re-raise to be handled by the Flask route's error handling
    except Exception as e: # Catch other potential pandas or file parsing errors
        logging.error(f"Error loading or parsing wardrobe data from '{WARDROBE_CSV_FILE}': {e}")
        raise 

    # 1. Get reference image descriptions for the uploaded image
    raw_descriptions: List[str] = []
    max_retries = 3 # Number of retries for generating descriptions
    for attempt in range(max_retries):
        try:
            # Note: image_resize is called within get_reference_image_description
            raw_descriptions = get_reference_image_description(image_path, STYLIST_PROMPT_GENERAL)
            if raw_descriptions: # If descriptions were successfully generated
                break
        except Exception as e:
            logging.error(f"Error in get_reference_image_description (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 == max_retries: # If this was the last attempt
                logging.error(f"Failed to generate descriptions for {image_path} after {max_retries} attempts.")
                raise # Re-raise the last exception
        logging.info(f"Retrying description generation for {image_path}, attempt {attempt + 2}...")
        time.sleep(1) # Wait a bit before retrying

    if not raw_descriptions:
        logging.info(f"No raw descriptions were generated for {image_path}.")
        return []

    # 2. Filter these descriptions for relevance
    filtered_descriptions = filter_image_descriptions(raw_descriptions, CLOTHING_CATEGORIES)
    if not filtered_descriptions:
        logging.info(f"No relevant clothing descriptions found after filtering for {image_path}.")
        return []

    # 3. Get similar items from the wardrobe for each relevant description
    for query_desc in filtered_descriptions:
        logging.debug(f"Querying wardrobe for description: '{query_desc}'")
        # Get top 1 match as per the original Flask app's logic for display
        recommendations = get_similar_text_from_query(
            query=query_desc,
            text_metadata_df=wardrobe_df,
            top_n=1 
        )
        if recommendations:
            # recommendations is Dict[int, Dict[str, Any]]
            # We want the image_uri from the first match (key 0, if it exists)
            first_match = recommendations.get(0) 
            if first_match and 'image_uri' in first_match:
                uri = first_match['image_uri']
                # Adjust URI for web display (remove 'static/' prefix if present)
                recommended_image_uris.append(uri.replace('static/', '', 1))
    
    # Return unique image URIs, preserving order of first appearance
    return list(dict.fromkeys(recommended_image_uris))

#-----------------------------------------
# Flask Routes
#-----------------------------------------
@app.route('/')
def home() -> str:
  """Serves the home page with the image upload form."""
  return render_template('index.html')

@app.route('/', methods=['POST'])
def upload() -> str: # Actually returns str (HTML content) or Werkzeug Response
  """Handles image uploads, processes them, and returns recommendations."""
  if 'file' not in request.files:
    logging.warning("File part missing in POST request.")
    return render_template('index.html', error_message="No file part in the request.")
  
  uploaded_file: FileStorage = request.files['file']
  
  if not uploaded_file or not uploaded_file.filename:
    logging.warning("No file selected by user for upload.")
    return render_template('index.html', error_message="No file selected.")

  if allowed_file(uploaded_file.filename):
    temp_filepath: Optional[str] = None 
    try:
      temp_filepath = save_uploaded_image(uploaded_file)
      logging.info(f"Uploaded file saved temporarily to: {temp_filepath}")
      
      recommended_images = get_recommendations_for_image(temp_filepath)
      logging.info(f"Recommendations for {uploaded_file.filename}: {recommended_images}")
      
      return render_template('index.html', recommended_images=recommended_images)

    except FileNotFoundError as e: 
        logging.error(f"Processing error due to missing file (likely wardrobe CSV): {e}")
        return render_template('index.html', error_message="Wardrobe data is missing. Please ensure the wardrobe has been processed.")
    except Exception as e:
      # Log the full exception for server-side diagnosis
      logging.exception(f"An unexpected error occurred during upload and processing of {uploaded_file.filename}: {e}")
      # Return a generic error message to the user
      return render_template('index.html', error_message="An internal error occurred while processing your image. Please try again later.")
    finally:
      # Clean up the temporarily saved uploaded file
      if temp_filepath and os.path.exists(temp_filepath):
        try:
            os.remove(temp_filepath)
            logging.info(f"Successfully removed temporary file: {temp_filepath}")
        except Exception as e_remove:
            logging.error(f"Error removing temporary file {temp_filepath}: {e_remove}")
  else:
    logging.warning(f"File type not allowed for filename: {uploaded_file.filename}. Allowed: {app.config['ALLOWED_EXTENSIONS']}")
    return render_template('index.html', error_message=f"File type not allowed. Please upload one of: {', '.join(app.config['ALLOWED_EXTENSIONS'])}.")


@app.errorhandler(500)
def server_error(e: Union[Exception, int]) -> Response: # Return type is a Flask Response
    """Custom error handler for internal server errors (HTTP 500)."""
    logging.exception(f'An unhandled server exception occurred: {e}')
    # It's good practice to return a rendered template for error pages.
    # The status code for the response will be 500 by default if not specified.
    return render_template('index.html', error_message="An unexpected server error occurred. We are looking into it!"), 500

#-----------------------------------------
# Main Execution Block (for running the Flask app)
#-----------------------------------------
if __name__ == "__main__":
    # Basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    # For production, consider more advanced logging (e.g., to a file, structured logging).
    
    # Port 8080 is a common alternative to port 80 (which requires root).
    # debug=False is crucial for production environments.
    app.run(host='0.0.0.0', port=8080, debug=False)
