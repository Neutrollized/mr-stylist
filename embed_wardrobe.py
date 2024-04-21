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


#-----------------------------------
# Initialize Vertex AI & Gemini
#-----------------------------------
PROJECT_ID = os.environ.get('MY_PROJECT_ID')  # @param {type:"string"}
LOCATION = "northamerica-northeast1"  # @param {type:"string"}

# if not running on colab, try to get the PROJECT_ID automatically
if "google.colab" not in sys.modules:
    import subprocess

    PROJECT_ID = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True
    ).strip()

#print(f"Your project ID is: {PROJECT_ID}")


vertexai.init(project=PROJECT_ID, location=LOCATION)

text_model = GenerativeModel("gemini-1.0-pro")
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")


#-----------------------------------------
# Helper Functions
#-----------------------------------------
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from helpers.image_utils import image_resize


# Use a more deterministic configuration with a low temperature
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
generation_config = GenerationConfig(
    temperature=0.0,		# higher = more creative (default 0.0)
    top_p=0.7,			# higher = more random responses, response drawn from more possible next tokens (default 0.95)
    top_k=20,			# higher = more random responses, sample from more possible next tokens (default 40)
    candidate_count=1,
    max_output_tokens=2048,
)


def generate_text(image_uri: str, prompt: str) -> str:
    # Query the model
    response = multimodal_model.generate_content(
        [
            # Add an example image
#            Part.from_uri(
#                "gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"
#            ),
            Part.from_image(Image.load_from_file(image_uri)),
            prompt,
        ],
        generation_config=generation_config,
    )
    #print(response)
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
    embeddings = text_embedding_model.get_embeddings([text])
    text_embedding = [embedding.values for embedding in embeddings][0]

    if return_array:
        text_embedding = np.fromiter(text_embedding, dtype=float)

    # returns 768 dimensional array
    return text_embedding


#-----------------------------------------
# Extract & store metadata of images
#-----------------------------------------
import glob
import pandas as pd
import time

from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as vision_model_Image
from vertexai.vision_models import MultiModalEmbeddingModel

text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding@001"
)


#image_metadata_df_json = pd.read_json('mywardrobe.json')
#print('DF from JSON')
#print(image_metadata_df_json)

# CSV more precise than JSON
# skipping first column as that's an additional column number
#image_metadata_df_csv = pd.read_csv('mywardrobe.csv', usecols=range(1, 5))
#print('DF from CSV')
#print(image_metadata_df_csv)


#image_description_prompt = """Describe the clothing shown in the image.
#Describe type, style, color and any designs on the clothing.
#"""

image_description_prompt = "Provide a few sentences describing the clothing's type, color, and style."

image_metadata_df = pd.DataFrame(columns=(
  'image_uri',
  'image_description_text',
  'image_description_text_embedding',
))

image_uri_path = 'images/'
image_count = 0

for image in list(glob.glob(image_uri_path + '/' + '*.JPG')):
  IMAGE = image_resize(image, 1024)
  print("processing: ", IMAGE)

  description_text = generate_text(
    image_uri=IMAGE,
    prompt=image_description_prompt,
  )
#  print(description_text)


  image_description_text_embedding = (
    get_text_embedding_from_text_embedding_model(text=description_text)
  )
  #print('IMAGE TEXT EMBEDDING')
  #print(image_description_text_embedding)
  #print(len(image_description_text_embedding))
  
  image_metadata_df.loc[image_count] = [IMAGE, description_text, image_description_text_embedding]
  image_count += 1
  time.sleep(3)		# to avoid hitting Gemini quota
 

image_metadata_df.to_csv('mywardrobe.csv')


print(image_metadata_df)
