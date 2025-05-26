# helpers/clothing_utils.py
"""
Utilities related to clothing categories and keyword matching.

This module defines constants for various clothing categories and provides
functions to help identify clothing items in textual descriptions based on keywords.
"""
from typing import List

# Define word lists for different clothing categories.
# These keywords are used to identify clothing types in text descriptions.
# Using uppercase for these constants as they are global configuration-like values.
HAT_WORD_LIST: List[str] = [' hat', ' cap', ' fedora', ' beanie']
JACKET_WORD_LIST: List[str] = [' jacket', ' coat', ' parka', ' blazer', ' vest']
SWEATER_WORD_LIST: List[str] = [' sweater', ' hoodie']
SHIRT_WORD_LIST: List[str] = [' t-shirt', ' shirt', ' tank top']
PANT_WORD_LIST: List[str] = [' pants', ' jeans', ' sweatpants', ' shorts', ' chinos', ' khakis', 'trousers']
SHOE_WORD_LIST: List[str] = [' shoes', ' sneakers', ' loafers', ' clogs']
# Consider adding:
# DRESS_WORD_LIST: List[str] = [' dress', ' gown', ' frock']
# SKIRT_WORD_LIST: List[str] = [' skirt', ' miniskirt']
# ACCESSORY_WORD_LIST: List[str] = [' belt', ' scarf', ' tie', ' handbag', ' glasses']

# List of all clothing category word lists.
# This list is used by functions that need to check against all defined categories.
CLOTHING_CATEGORIES: List[List[str]] = [
    HAT_WORD_LIST,
    JACKET_WORD_LIST,
    SWEATER_WORD_LIST,
    SHIRT_WORD_LIST,
    PANT_WORD_LIST,
    SHOE_WORD_LIST,
    # Add other category lists here if defined above
]

def any_list_element_in_string(input_list_of_lists: List[List[str]], input_string: str) -> int:
  """
  Checks if any element from any of the inner lists (clothing categories) is present in the input string.

  The comparison is case-insensitive.

  Args:
      input_list_of_lists: A list of lists, where each inner list contains keyword strings
                           (e.g., `CLOTHING_CATEGORIES`).
      input_string: The string to check for the presence of these keywords.

  Returns:
      The number of distinct inner lists (categories) that have at least one keyword
      present in the `input_string`. Returns 0 if `input_string` is empty or no
      keywords from any category are found.
  """
  if not input_string:
    return 0

  match_count = 0
  input_string_lower = input_string.lower() # For case-insensitive matching

  for word_list in input_list_of_lists:
    # Check if any word from the current word_list is in input_string_lower
    if any(word.lower() in input_string_lower for word in word_list):
      match_count += 1
      # No 'break' here if we want to count all categories that match.
      # The original logic had a 'break', which means it counted a category
      # if *any* word from that category matched, then moved to the next category.
      # The current implementation (and the original one after the loop fix)
      # counts how many *categories* have at least one match.
      # If the intention was to count total *word* matches, the logic would be different.
      # The docstring implies counting categories with matches.
      # The original break was inside the inner loop, meaning after finding one word from a list,
      # it would increment match_count and break from checking other words *in that same list*.
      # This is correct for "number of inner lists that have at least one word present".
  return match_count
