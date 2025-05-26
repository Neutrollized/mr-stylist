# helpers/recommender_utils.py
"""
Utility functions supporting the recommendation generation process.

This module includes functions for calculating similarity scores (e.g., cosine similarity)
and for displaying recommendation results, particularly in a command-line interface (CLI) context.
"""
import logging
from PIL import Image as PIL_Image
import pandas as pd
import numpy as np
from numpy.linalg import norm # For vector norm calculation

# Typing imports
from typing import Dict, Any, List # Added List for type hinting

def get_cosine_score(
    dataframe_row: pd.Series, column_name: str, query_embedding: np.ndarray
) -> float:
    """
    Calculates the cosine similarity between a query embedding and an embedding from a DataFrame row.

    Args:
        dataframe_row: A pandas Series representing a single row from the DataFrame.
                       It's expected that `dataframe_row[column_name]` contains the embedding
                       as a list of numerical values (floats or ints).
        column_name: The name of the column in `dataframe_row` that holds the embedding.
        query_embedding: A NumPy array representing the query embedding.

    Returns:
        The cosine similarity score as a float. Returns 0.0 if an error occurs
        during calculation (e.g., incompatible types, math error).
    """
    try:
        # Assuming dataframe_row[column_name] is already a list of numbers (floats/ints)
        # due to prior conversion during CSV loading.
        row_embedding_list: List[float] = dataframe_row[column_name]
        
        if not isinstance(row_embedding_list, list) or not all(isinstance(x, (int, float)) for x in row_embedding_list):
            logging.warning(f"Row embedding in column '{column_name}' is not a list of numbers. Value: {row_embedding_list}")
            return 0.0

        row_embedding_np = np.array(row_embedding_list, dtype=float)

        # Ensure embeddings are 1-D arrays (vectors)
        if row_embedding_np.ndim != 1 or query_embedding.ndim != 1:
            logging.warning("Embeddings must be 1-D arrays for cosine similarity.")
            return 0.0
        
        if row_embedding_np.shape != query_embedding.shape:
            logging.warning(f"Embedding shapes do not match: Row {row_embedding_np.shape}, Query {query_embedding.shape}")
            return 0.0

        # Calculate cosine similarity
        # np.dot(a, b) / (norm(a) * norm(b))
        similarity_score = np.dot(row_embedding_np, query_embedding) / (norm(row_embedding_np) * norm(query_embedding))
        
        # Handle potential NaN or Inf values if norms are zero (though norm() should handle zero vectors)
        if np.isnan(similarity_score) or np.isinf(similarity_score):
            logging.warning(f"Cosine similarity resulted in NaN or Inf. Embeddings might be zero vectors. Row: {row_embedding_np}, Query: {query_embedding}")
            return 0.0
            
        return float(similarity_score)

    except TypeError as te:
        logging.error(f"TypeError during cosine similarity calculation for column '{column_name}': {te}. Row data: {dataframe_row.get(column_name)}")
        return 0.0
    except ValueError as ve:
        logging.error(f"ValueError during cosine similarity calculation for column '{column_name}': {ve}. Row data: {dataframe_row.get(column_name)}")
        return 0.0
    except Exception as e:
        logging.error(f"Unexpected error in get_cosine_score for column '{column_name}': {e}. Row data: {dataframe_row.get(column_name)}")
        return 0.0


def show_filter_results(results: Dict[int, Dict[str, Any]]):
  """
  Displays images of the recommended items and their details (cosine score, URI).
  This function is intended for CLI usage where images can be shown directly.

  Args:
      results: A dictionary where keys are integers and values are dictionaries
               containing 'cosine_score', 'image_uri', and 'image_description_text'.
               Example: {0: {'cosine_score': 0.9, 'image_uri': 'path/to/img.jpg', ...}, ...}

  Returns:
      None
  """
  if not results: # Simplified check for empty dictionary
      print("--> NO GOOD MATCH FOUND IN YOUR WARDROBE <--")
  else:
    for item_key in sorted(results.keys()): # Sort by key for consistent order
      item_details = results[item_key]
      cosine_score = item_details.get('cosine_score', 'N/A')
      image_uri = item_details.get('image_uri')
      # image_description = item_details.get('image_description_text', 'No description.') # Uncomment if needed

      print(f"\nMatch Rank: {item_key + 1}") # Assuming keys are 0-indexed ranks
      print(f"  Cosine Score: {cosine_score:.4f}") # Format score
      print(f"  Image URI: {image_uri}")
      # print(f"  Description: {image_description}") # Uncomment if needed
      
      if image_uri:
        try:
          img = PIL_Image.open(image_uri)
          img.show() # This will open the image using the default system image viewer
        except FileNotFoundError:
          logging.error(f"Image file not found at URI: {image_uri}")
          print(f"  Error: Image file not found at {image_uri}")
        except Exception as e:
          logging.error(f"Error opening or showing image {image_uri}: {e}")
          print(f"  Error: Could not display image {image_uri}. Details: {e}")
      else:
        print("  Image URI not provided.")

