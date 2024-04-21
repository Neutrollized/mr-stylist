from PIL import Image as PIL_Image

import pandas as pd
import numpy as np
from numpy.linalg import norm

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
    #print(float_list)

    #text_cosine_score = round(np.dot(dataframe[column_name], input_text_embd), 2)
    #text_cosine_score = round(np.dot(float_list, input_text_embd), 2)
    text_cosine_score = np.dot(float_list, input_text_embd)/(norm(float_list)*norm(input_text_embd))
    #print('TEXT_COSINE_SCORE', text_cosine_score)

    return text_cosine_score



def filter_results(results: dict):
  for item in results:
    #print("item:", item)
    image_uri=results[item]['image_uri']
    print("image_uri:", image_uri)
    image=PIL_Image.open(image_uri)
    image.show()
