from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps

def image_resize(filename: str, max_size: int) -> str:
  MAX_SIZE = (max_size, max_size)
  img = PIL_Image.open(filename)
  img = PIL_ImageOps.exif_transpose(img)
  img.thumbnail(MAX_SIZE, PIL_Image.Resampling.LANCZOS)
  img.save(filename, format='JPEG', quality=100)
  return filename
