#! /usr/bin/env python3
"""
Outfit Recommender CLI Tool.

This script provides a command-line interface (CLI) to recommend outfits from a
pre-processed wardrobe based on an input image. It leverages Vertex AI for image
description generation and text embeddings to find similar items.

Workflow:
1. Initializes Vertex AI services.
2. Loads the wardrobe data (embeddings and metadata) from a CSV file.
3. Takes an input image path and (optionally) the number of recommendations per description.
4. Generates textual descriptions for the input image using a multimodal AI model.
5. Filters these descriptions to keep only those relevant to defined clothing categories.
6. For each relevant description, queries the wardrobe data to find similar items based on
   embedding cosine similarity.
7. Displays the top N recommended items from the wardrobe, showing their images and details.
"""
import os
import sys
import logging # For logging status and errors
import time # For retry delays
from typing import Any, Dict, List

import pandas as pd
import numpy as np # For potential use in imported functions

# Config imports
from config import (
    COSINE_SCORE_THRESHOLD, # Used by get_similar_text_from_query
    STYLIST_PROMPT_GENERAL, # Default prompt for generating image descriptions
    WARDROBE_CSV_FILE,      # Path to the wardrobe data CSV
)

# Helper imports
from helpers.image_utils import image_resize
from helpers.recommender_utils import get_cosine_score, show_filter_results
from helpers.clothing_utils import CLOTHING_CATEGORIES, any_list_element_in_string

# Vertex AI utilities
from vertex_ai_utils import (
    init_vertex_ai,
    generate_text,
    get_text_embedding_from_text_embedding_model,
)

#-----------------------------------
# Core Recommender Functions
#-----------------------------------

def get_reference_image_description(image_filename: str, stylist_prompt: str) -> List[str]:
    """
    Generates textual descriptions for a given image using a stylist prompt.

    The image is first resized. Then, a multimodal model generates descriptions,
    which are subsequently split into a list.

    Args:
        image_filename: Path to the input image file.
        stylist_prompt: The prompt to guide the AI model in generating descriptions.

    Returns:
        A list of generated textual descriptions for the image.
        Returns an empty list if description generation fails or the response is empty.
        
    Raises:
        Propagates exceptions from `image_resize` or `generate_text` if they occur.
    """
    logging.info(f"Resizing image: {image_filename}")
    resized_image_file = image_resize(image_filename, 1280) # Standard resize dimension
    
    logging.info(f"Generating descriptions for: {resized_image_file} using prompt: '{stylist_prompt[:50]}...'")
    response_text = generate_text(
        image_uri=resized_image_file,
        prompt=stylist_prompt,
        # System instruction is handled by the shared model in vertex_ai_utils
    )
    
    # Split descriptions, which might be separated by double newlines or just newlines.
    # Filter out empty strings that might result from splitting.
    if response_text:
        output_descriptions = [desc.strip() for desc in response_text.split('\n\n') if desc.strip()]
        if not output_descriptions: # If split by \n\n failed or resulted in no content, try single \n
            output_descriptions = [desc.strip() for desc in response_text.split('\n') if desc.strip()]
        logging.debug(f"Generated descriptions for {image_filename}: {output_descriptions}")
        return output_descriptions
    else:
        logging.warning(f"No response text from generate_text for {image_filename}.")
        return []

def filter_image_descriptions(descriptions: List[str], clothing_categories: List[List[str]]) -> List[str]:
    """
    Filters a list of image descriptions to keep only those relevant to specified clothing categories.

    Args:
        descriptions: A list of textual descriptions.
        clothing_categories: A list of lists, where each inner list contains keywords for a clothing category.

    Returns:
        A list of descriptions that contain keywords from at least one clothing category.
    """
    valid_queries: List[str] = []
    if not descriptions:
        logging.debug("No descriptions provided to filter.")
        return valid_queries

    for desc in descriptions:
        num_clothing_types = any_list_element_in_string(clothing_categories, desc)
        logging.debug(f"Clothing types found in description '{desc}': {num_clothing_types}")
        if num_clothing_types > 0: # Keep if it mentions at least one clothing type
            valid_queries.append(desc)
        else:
            logging.info(f"Description removed (no relevant categories): '{desc}'")
    return valid_queries

def get_similar_text_from_query(
    query: str,
    text_metadata_df: pd.DataFrame,
    column_name: str = "image_description_text_embedding", # Default column for embeddings
    top_n: int = 3
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar items from a wardrobe DataFrame based on a query description.

    Similarity is determined by cosine similarity between the query embedding and
    pre-calculated embeddings in the DataFrame.

    Args:
        query: The textual query (e.g., a filtered image description).
        text_metadata_df: Pandas DataFrame containing wardrobe item metadata, including embeddings.
                          Expected to have columns 'image_uri', 'image_description_text', and `column_name`.
        column_name: The name of the column in `text_metadata_df` that stores the embeddings.
        top_n: The maximum number of similar items to return.

    Returns:
        A dictionary where keys are 0-indexed ranks and values are dictionaries
        containing details of the matched items ('image_uri', 'image_description_text', 'cosine_score').
        Returns an empty dictionary if no items meet the similarity threshold or if errors occur.
        
    Raises:
        KeyError: If the specified `column_name` (for embeddings) or other expected columns
                  like 'image_uri', 'image_description_text' are not in `text_metadata_df`.
    """
    if column_name not in text_metadata_df.columns:
        logging.error(f"Embedding column '{column_name}' not found in DataFrame.")
        raise KeyError(f"Column '{column_name}' not found in the DataFrame.")
    if not all(col in text_metadata_df.columns for col in ['image_uri', 'image_description_text']):
        logging.error("DataFrame is missing required columns: 'image_uri' or 'image_description_text'.")
        raise KeyError("DataFrame is missing required columns: 'image_uri' or 'image_description_text'.")

    logging.debug(f"Generating embedding for query: '{query[:100]}...'")
    # Assuming get_text_embedding_from_text_embedding_model returns List[float] by default
    query_embedding_list = get_text_embedding_from_text_embedding_model(text=query, return_array=False)
    if not query_embedding_list: # Check if embedding generation failed
        logging.warning(f"Failed to generate embedding for query: {query}")
        return {}
    query_vector = np.array(query_embedding_list, dtype=float)


    logging.debug("Calculating cosine scores against wardrobe data...")
    cosine_scores = text_metadata_df.apply(
        lambda row: get_cosine_score(
            row, # Pass the entire row (Series)
            column_name,
            query_vector,
        ),
        axis=1, # Apply per row
    )

    # Get top N cosine scores that are also above or equal to the threshold
    relevant_indices = cosine_scores[cosine_scores >= COSINE_SCORE_THRESHOLD].nlargest(top_n).index
    
    final_results: Dict[int, Dict[str, Any]] = {}
    for i, index in enumerate(relevant_indices):
        final_results[i] = {
            "image_uri": text_metadata_df.loc[index, "image_uri"],
            "image_description_text": text_metadata_df.loc[index, "image_description_text"],
            "cosine_score": cosine_scores[index], # Use the actual score from the Series
        }
    
    if not final_results:
        logging.info(f"No items found meeting similarity criteria for query: '{query[:100]}...'")
    return final_results

#-----------------------------------------
# CLI Workflow Orchestration
#-----------------------------------------
def recommend_outfits_cli(image_path: str, num_recommendations: int) -> None:
    """
    Orchestrates the command-line interface workflow for outfit recommendations.

    Loads wardrobe data, processes the input image to get descriptions, filters them,
    queries for similar items, and displays the results.

    Args:
        image_path: Path to the input image for which to recommend outfits.
        num_recommendations: The number of recommendations to show for each relevant description.
    """
    try:
        # Ensure embeddings are loaded as lists of floats
        wardrobe_df = pd.read_csv(
            WARDROBE_CSV_FILE,
            converters={
                "image_description_text_embedding": lambda x: [float(i) for i in x.strip("[]").split(",")]
            }
        )
        logging.info(f"Successfully loaded wardrobe data from '{WARDROBE_CSV_FILE}'.")
    except FileNotFoundError:
        # Using print for direct CLI feedback on critical errors
        print(f"ERROR: Wardrobe data file '{WARDROBE_CSV_FILE}' not found. Please run embed_wardrobe.py first.")
        return
    except Exception as e:
        print(f"ERROR: Could not load or parse wardrobe data from '{WARDROBE_CSV_FILE}'. Details: {e}")
        return

    if wardrobe_df.empty:
        print(f"WARNING: Wardrobe data ('{WARDROBE_CSV_FILE}') is empty. No recommendations possible.")
        return

    logging.info(f"Processing image for CLI recommendation: {image_path}")
    
    raw_descriptions: List[str] = []
    retry_count = 0
    max_retries = 3 # Max attempts to get image descriptions

    while retry_count < max_retries:
        try:
            raw_descriptions = get_reference_image_description(image_path, STYLIST_PROMPT_GENERAL)
            if raw_descriptions: # If successful, exit retry loop
                break
        except Exception as e:
            logging.error(f"Error getting reference image description (attempt {retry_count + 1}/{max_retries}): {e}")
        
        retry_count += 1
        if retry_count < max_retries:
            logging.info(f"Retrying reference image description generation (attempt {retry_count + 1}/{max_retries})...")
            time.sleep(1) # Brief pause before retrying
            
    if not raw_descriptions:
        print(f"Could not generate reference descriptions for '{image_path}' after {max_retries} attempts.")
        return

    filtered_descriptions = filter_image_descriptions(raw_descriptions, CLOTHING_CATEGORIES)

    if not filtered_descriptions:
        print("No relevant clothing descriptions could be extracted from the input image after filtering.")
        return

    # User-facing CLI output starts here
    print(f"\n=== Recommendations for {os.path.basename(image_path)} ===")
    for i, query_desc in enumerate(filtered_descriptions):
        print(f"\n--- Based on your item's description: \"{query_desc}\" ---")
        
        try:
            recommendations = get_similar_text_from_query(
                query=query_desc,
                text_metadata_df=wardrobe_df,
                top_n=num_recommendations
            )
        except KeyError as e: # Catch if expected columns are missing from DataFrame
            print(f"Error: DataFrame is missing an expected column. Details: {e}")
            # Stop further processing for this description if DataFrame structure is wrong
            continue 
            
        if recommendations:
            # show_filter_results is designed for CLI and uses print for user output
            show_filter_results(recommendations) 
        else:
            print("No matching items found in your wardrobe for this particular description.")

#-----------------------------------------
# Main Execution Block (CLI Entry Point)
#-----------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    init_vertex_ai() # Initialize Vertex AI services

    if len(sys.argv) < 2:
        # Using print for CLI usage instructions
        print("Usage: python recommender.py <image_path> [num_recommendations]")
        print("Example: python recommender.py static/images/input_image.jpg 3")
        sys.exit(1)

    cli_image_path = sys.argv[1]
    # Default to 1 recommendation if not specified
    cli_num_recommendations = int(sys.argv[2]) if len(sys.argv) > 2 else 1 

    if not os.path.exists(cli_image_path):
        print(f"ERROR: Image path '{cli_image_path}' does not exist.")
        sys.exit(1)
        
    recommend_outfits_cli(cli_image_path, cli_num_recommendations)
