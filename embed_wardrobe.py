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


#-----------------------------------
# Variables
#-----------------------------------
GEMINI_MODEL="gemini-2.0-flash-001"
EMBEDDING_MODEL="text-embedding-005"
TEMPERATURE=0.1
TOP_P=0.8
TOP_K=25
STYLIST_PROMPT="Provide a few sentences describing the clothing's type, color, and style"


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

print(f"Your project ID is: {PROJECT_ID}")


vertexai.init(project=PROJECT_ID, location=LOCATION)

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
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from helpers.image_utils import image_resize


# Use a more deterministic configuration with a low temperature
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
generation_config = GenerationConfig(
    temperature=TEMPERATURE,		# higher = more creative (default 0.0)
    top_p=TOP_P,			# higher = more random responses, response drawn from more possible next tokens (default 0.95)
    top_k=TOP_K,			# higher = more random responses, sample from more possible next tokens (default 40)
    candidate_count=1,
    max_output_tokens=2048,
)


def generate_text(image_uri: str, prompt: str) -> str:
    # Query the model
    response = multimodal_model.generate_content(
        [
            Part.from_image(Image.load_from_file(image_uri)),
            prompt,
        ],
        generation_config=generation_config,
    )
    return response.text


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

    return text_embedding


#-----------------------------------------
# Extract & store metadata of images
#-----------------------------------------
import glob
import pandas as pd
import time

image_description_prompt=STYLIST_PROMPT

image_metadata_df = pd.DataFrame(columns=(
  'image_uri',
  'image_description_text',
  'image_description_text_embedding',
))

image_uri_path = 'static/images/'
image_count = 0

for image in list(glob.glob(image_uri_path + '/' + '*.JPG')):
  IMAGE = image_resize(image, 1024)
  print("processing: ", IMAGE)

  description_text = generate_text(
    image_uri=IMAGE,
    prompt=image_description_prompt,
  )

  image_description_text_embedding = (
    get_text_embedding_from_text_embedding_model(text=description_text)
  )
  
  image_metadata_df.loc[image_count] = [IMAGE, description_text, image_description_text_embedding]
  image_count += 1
  time.sleep(8)		# to avoid hitting Gemini quota
 

image_metadata_df.to_csv('mywardrobe.csv')

print(image_metadata_df)
