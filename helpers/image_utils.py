from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps

def image_resize(filename: str, max_size: int) -> str:
  """
  Resize image

  Args:
      filename: filename of file to resize
      max_size: integer, max size the length or heigh of image can reach

  Returns:
      The filename of the resized image
  """

  MAX_SIZE = (max_size, max_size)
  img = PIL_Image.open(filename)
  img = PIL_ImageOps.exif_transpose(img)
  img.thumbnail(MAX_SIZE, PIL_Image.Resampling.LANCZOS)
  img.save(filename, format='JPEG', quality=100)
  return filename


#-----------
# WIP
#-----------
def gemini_model_text_embed(text: str) -> list[float]:
    embedding = genai.embed_content(model="models/text-embedding-004",
                                    content=text,
                                    task_type="retrieval_query")

    return embedding["embedding"]
