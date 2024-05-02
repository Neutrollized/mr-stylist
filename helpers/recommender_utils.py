from PIL import Image as PIL_Image

import pandas as pd
import numpy as np
from numpy.linalg import norm


def any_list_element_in_string(clothing_list: list, string: str) -> int:
  """
  This function checks if any element from list is present in the given string.

  Args:
      clothing_list: A list of elements (also lists) to search for.
      string: The string to search within.

  Returns:
      int: the number of different clothing types in string
  """

  list_membership_count = 0
  # Iterate through each element in list1
  for list in clothing_list:
    # Check if the element is present in the string (case-sensitive)
    if any(element in string for element in list):
      list_membership_count += 1

  # If no element is found, return False
  return list_membership_count



def get_cosine_score(
    dataframe: pd.DataFrame, column_name: str, input_text_embd: np.ndarray
) -> float:
    """
    Calculates the cosine similarity between the user query embedding and the dataframe embedding for a specific column.

    Args:
        dataframe: The pandas DataFrame containing the data to compare against.
        column_name: The name of the column containing the embeddings to compare with.
        input_text_embd: The NumPy array representing the user query embedding.

    Returns:
        The cosine similarity score (rounded to two decimal places) between the user query embedding and the dataframe embedding.
    """

    float_list = [float(i) for i in dataframe[column_name]]
    text_cosine_score = np.dot(float_list, input_text_embd)/(norm(float_list)*norm(input_text_embd))

    return text_cosine_score



def show_filter_results(results: dict):
  """
  Displays image of the returned results

  Args:
      results: dictionary (JSON) of clothing in catalogue/wardrobe

  Returns:
      None
  """

  for item in results:
    image_uri=results[item]['image_uri']
    print("image_uri:", image_uri)
    image=PIL_Image.open(image_uri)
    image.show()

