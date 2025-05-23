#! /usr/bin/env python3

import os, sys

import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part,
)

from google import genai
from google.genai.types import EmbedContentConfig

from helpers.image_utils import image_resize
from helpers.recommender_utils import any_list_element_in_string
from helpers.recommender_utils import get_cosine_score

#-----------------------------------
# Variables
#-----------------------------------
GEMINI_MODEL="gemini-2.0-flash-001"
EMBEDDING_MODEL="text-embedding-005"
COSINE_SCORE_THRESHOLD=0.65
TEMPERATURE=0.1
TOP_P=0.8
TOP_K=25
#STYLIST_PROMPT="Describe the clothes in the photo.  Description should only include clothing 'type', 'color', 'style', and 'design'. Make sure to only describe each individual article of clothing, and give a separate response."
STYLIST_PROMPT="Can you describe the clothes in the photo, including style, color, and any designs?  Make sure to only describe each individual article of clothing, and give a separate response."


#-----------------------------------
# Initialize Vertex AI & Gemini
#-----------------------------------
PROJECT_ID = os.environ.get('MY_PROJECT_ID')  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# if not running on colab, try to get the PROJECT_ID automatically
if "google.colab" not in sys.modules:
    import subprocess

    PROJECT_ID = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True
    ).strip()


vertexai.init(project=PROJECT_ID, location=LOCATION)

#multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
multimodal_model = GenerativeModel(
        GEMINI_MODEL,
        system_instruction=[
            "You are a fashion stylist.",
            "Your mission is to describe the clothing you see.",
        ],
)

#-----------------------------------------
# Helper Functions
#-----------------------------------------
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

def generate_text(image_uri: str, prompt: str) -> str:
    # Query the model
    response = multimodal_model.generate_content(
        [
            Part.from_image(Image.load_from_file(image_uri)),
            prompt,
        ]
    )
    return response.text


def get_image_embedding_from_multimodal_embedding_model(
    image_uri: str,
    embedding_size: int = 512,
    text: Optional[str] = None,
    return_array: Optional[bool] = False,
) -> list:
    """Extracts an image embedding from a multimodal embedding model.
    The function can optionally utilize contextual text to refine the embedding.

    Args:
        image_uri (str): The URI (Uniform Resource Identifier) of the image to process.
        text (Optional[str]): Optional contextual text to guide the embedding generation. Defaults to "".
        embedding_size (int): The desired dimensionality of the output embedding. Defaults to 512.
        return_array (Optional[bool]): If True, returns the embedding as a NumPy array.
        Otherwise, returns a list. Defaults to False.

    Returns:
        list: A list containing the image embedding values. If `return_array` is True, returns a NumPy array instead.
    """
    image = vision_model_Image.load_from_file(image_uri)
    embeddings = multimodal_embedding_model.get_embeddings(
        image=image, contextual_text=text, dimension=embedding_size
    )  # 128, 256, 512, 1408
    image_embedding = embeddings.image_embedding

    if return_array:
        image_embedding = np.fromiter(image_embedding, dtype=float)

    return image_embedding


def get_text_embedding_from_text_embedding_model(
    text: str,
    return_array: Optional[bool] = False,
) -> list:
    """
    Generates a numerical text embedding from a provided text input using a text embedding model.

    Args:
        text: The input text string to be embedded.
        return_array: If True, returns the embedding as a NumPy array.
                      If False, returns the embedding as a list. (Default: False)

    Returns:
        list or numpy.ndarray: A 768-dimensional vector representation of the input text.
                               The format (list or NumPy array) depends on the
                               value of the 'return_array' parameter.
    """
    client = genai.Client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",  # Optional
            output_dimensionality=768,  # Optional
        ),
    )

    text_embedding = response.embeddings[0].values

    if return_array:
        text_embedding = np.fromiter(text_embedding, dtype=float)

    # returns 768 dimensional array
    return text_embedding


def get_similar_text_from_query(
    query: str,
    text_metadata_df: pd.DataFrame,
    column_name: str = "",
    top_n: int = 3,
    chunk_text: bool = True,
    print_citation: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar text passages from a metadata DataFrame based on a text query.

    Args:
        query: The text query used for finding similar passages.
        text_metadata_df: A Pandas DataFrame containing the text metadata to search.
        column_name: The column name in the text_metadata_df containing the text embeddings or text itself.
        top_n: The number of most similar text passages to return.
        embedding_size: The dimensionality of the text embeddings (only used if text embeddings are stored in the column specified by `column_name`).
        chunk_text: Whether to return individual text chunks (True) or the entire page text (False).
        print_citation: Whether to immediately print formatted citations for the matched text passages (True) or just return the dictionary (False).

    Returns:
        A dictionary containing information about the top N most similar text passages, including cosine scores, page numbers, chunk numbers (optional), and chunk text or page text (depending on `chunk_text`).

    Raises:
        KeyError: If the specified `column_name` is not present in the `text_metadata_df`.
    """

    if column_name not in text_metadata_df.columns:
        raise KeyError(f"Column '{column_name}' not found in the 'text_metadata_df'")

    query_vector = get_text_embedding_from_text_embedding_model(text=query)

    # Calculate cosine similarity between query text and metadata text
    cosine_scores = text_metadata_df.apply(
        lambda row: get_cosine_score(
            row,
            column_name,
            query_vector,
        ),
        axis=1,
    )

    # Get top N cosine scores and their indices
    top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
    top_n_scores = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched text and their information
    final_text: Dict[int, Dict[str, Any]] = {}

    for matched_textno, index in enumerate(top_n_indices):
        # Create a sub-dictionary for each matched text
        final_text[matched_textno] = {}

        # Store page number
        final_text[matched_textno]["image_uri"] = text_metadata_df.iloc[index][
            "image_uri"
        ]

        # Store page number
        final_text[matched_textno]["image_description_text"] = text_metadata_df.iloc[index][
            "image_description_text"
        ]

        # Store cosine score
        final_text[matched_textno]["cosine_score"] = top_n_scores[matched_textno]

    # Optionally print citations immediately
    if print_citation:
        print_text_to_text_citation(final_text, chunk_text=chunk_text)

    return final_text


def get_user_query_text_embeddings(user_query: str) -> np.ndarray:
    """
    Extracts text embeddings for the user query using a text embedding model.

    Args:
        user_query: The user query text.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query text embedding.
    """
    
    print(user_query)

    return get_text_embedding_from_text_embedding_model(user_query)


#-----------------------------------------
# Extract & store metadata of images
#-----------------------------------------

def get_reference_image_description(image_filename: str) -> list:
  # Use a more deterministic configuration with a low temperature
  generation_config = GenerationConfig(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    candidate_count=1,	# reponse
    max_output_tokens=512,
  )

  IMAGE_FILE = image_resize(image_filename, 1280)
  image = Image.load_from_file(IMAGE_FILE)

  response = multimodal_model.generate_content(
    [
        #"Can you describe the clothes in the photo, including style, color, and any designs?  Make sure to only describe each individual article of clothing, and give a separate response.",
        "Describe the clothes in the photo.  Description should only include clothing 'type', 'color', 'style', and 'design'. Make sure to only describe each individual article of clothing, and give a separate response.",
        image
    ],
    generation_config=generation_config
  )

  output_description_text = response.text.split('\n\n')
  return output_description_text


#-----------------------------------------
# Extract & store metadata of images
#-----------------------------------------
import glob, sys
import pprint

from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as vision_model_Image
from vertexai.vision_models import MultiModalEmbeddingModel

# for embedding
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

image_metadata_df_csv = pd.read_csv("mywardrobe.csv",converters={"image_description_text_embedding": lambda x: x.strip("[]").split(", ")})


#--------------
# Routes
#--------------
from flask import Flask, render_template, request, redirect
from tempfile import NamedTemporaryFile
import os, logging

app = Flask(__name__, static_url_path='')

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_image(image_data, extension="JPG"):
  """
  Saves uploaded image data to a temporary file.

  Args:
      image_data: Bytes representing the image data.
      extension (str, optional): The file extension for the temporary file. Defaults to "jpg".

  Returns:
      str: The path to the temporary file containing the image.
  """
  with NamedTemporaryFile(delete=False, suffix="." + extension) as temp_file:
    temp_file.write(image_data)
    return temp_file.name

@app.route('/')
def home():
  return render_template('index.html')

UPLOAD_FOLDER = 'temp/'
@app.route('/', methods=['POST'])
def upload() -> str:
  uploaded_file = request.files['file']
  
  if not allowed_file(uploaded_file.filename):
    return render_template('index.html')

  if not uploaded_file:
    return 'No file uploaded.', 400

  try:
    os.mkdir(UPLOAD_FOLDER)
  except FileExistsError:
    pass

  uploaded_file.save(os.path.join(UPLOAD_FOLDER, uploaded_file.filename))

  hat_word_list=[' hat', ' cap', ' fedora', ' beanie']
  jacket_word_list=[' jacket', ' coat', ' parka', ' blazer', ' vest']
  sweater_word_list=[' sweater', ' hoodie']
  shirt_word_list=[' t-shirt', ' shirt', ' tank top']
  pant_word_list=[' pants', ' jeans', ' sweatpants', ' shorts', ' chinos', ' khakis', 'trousers']
  shoe_word_list=[' shoes', ' sneakers', ' loafers', ' clogs']
  clothing_list=[hat_word_list, jacket_word_list, sweater_word_list, shirt_word_list, pant_word_list, shoe_word_list]

  # add a retry for generating descriptions
  # try to ensure it generates separate descriptions for each article of clothing
  retry_count = 0
  while retry_count < 5:
    queries = get_reference_image_description(UPLOAD_FOLDER + uploaded_file.filename)

    # filter out responses that have more than 1 clothing type listed
    for query in queries:
      num_clothing_types=any_list_element_in_string(clothing_list, query)
      if num_clothing_types == 0:
        queries.remove(query)

    if len(queries) == 0:
      retry_count += 1
    else:
      break


  item_num = 0
  recommended_images=[]
  for query in queries: 
    find_match = get_similar_text_from_query(
      query,
      image_metadata_df_csv,
      column_name = "image_description_text_embedding",
      top_n=1,
      chunk_text = False,
    )
    item_num +=1
    image_uri=find_match[0]['image_uri']
    # strip the 'static/' the filepath
    recommended_images.append(image_uri.replace('static/', ''))

  os.remove(UPLOAD_FOLDER + uploaded_file.filename)
  return render_template('index.html', recommended_images=recommended_images)



@app.errorhandler(500)
def server_error(e: Union[Exception, int]) -> str:
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


#--------------
# Main
#--------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='80')
