# helpers/image_utils.py
"""
Image processing utilities for the Fashion Recommendation Application.

This module provides functions for common image manipulation tasks such as
resizing images. It uses the Pillow (PIL) library.
"""
import logging
import os # Already imported at the end, moving it to the top for convention.
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps

def image_resize(filename: str, max_size: int) -> str:
  """
  Resizes an image file in place to fit within a specified maximum dimension (width or height),
  preserving aspect ratio. Handles image orientation based on EXIF data.
  The image is saved in JPEG format.

  Args:
      filename: The path to the image file to resize.
      max_size: The maximum size (width or height) the image should have after resizing.

  Returns:
      The filename of the resized image if successful.
      
  Raises:
      FileNotFoundError: If the image file does not exist.
      IOError: If the file cannot be opened, is not a valid image, or during saving.
      Exception: For other PIL-related processing errors.
  """
  if not os.path.exists(filename):
      logging.error(f"Image file not found: {filename}")
      raise FileNotFoundError(f"Image file not found: {filename}")

  try:
    img = PIL_Image.open(filename)
    
    # Handle EXIF orientation
    img = PIL_ImageOps.exif_transpose(img)
    
    # Convert to RGB if it's a GIF or has other modes like P (palette) or RGBA
    # This is important for saving as JPEG, which doesn't support transparency well.
    if img.mode in ("RGBA", "P", "LA") or (img.format == "GIF"):
        # Create a new image with a white background if the original has transparency
        # Using alpha composite with a white background for RGBA/LA
        if img.mode in ("RGBA", "LA"):
            background = PIL_Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1]) # Paste using alpha channel as mask
            img = background
        else: # For P mode (palette-based) or other direct conversions
            img = img.convert("RGB")

    original_format = img.format # Store original format for logging, if needed

    max_dimensions = (max_size, max_size)
    img.thumbnail(max_dimensions, PIL_Image.Resampling.LANCZOS)
    
    # Save the image, overwriting the original file
    # quality=100 might be too high for general use, 85-95 is often a good balance.
    # For this project, we'll stick to 100 as per original.
    img.save(filename, format='JPEG', quality=100)
    logging.info(f"Image '{filename}' (original format: {original_format}) resized and saved as JPEG.")
    
    return filename
  
  except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
      logging.error(f"File not found during PIL processing (should have been caught earlier): {filename}")
      raise
  except IOError as e:
      logging.error(f"IOError processing image {filename} (e.g., invalid image, save error): {e}")
      raise
  except Exception as e:
      logging.error(f"Unexpected error processing image {filename}: {e}")
      raise
