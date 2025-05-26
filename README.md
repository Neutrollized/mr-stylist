# MR Stylist - Multimodal RAG Fashion Recommender

[**M**]ultimodal [**R**]AG **Stylist** is a prototype clothing recommender built using Gemini's multimodal capabilities and a Retrieval-Augmented Generation (RAG)-like approach to find similar clothing items from a user's wardrobe.

This prototype can be adapted for other domains like furniture/interior design recommendations or even suggesting recipe alternatives based on available groceries.

`v0.4.0` (post-refactor) has been updated for better code organization and maintainability. A demo of an earlier version (`v0.3.0`) is available on [YouTube](https://www.youtube.com/watch?v=g3WcuO87FUI).

## Code Organization

The project has been refactored for improved modularity:
-   **`config.py`**: Centralizes all shared configurations, such as AI model names, file paths, and model parameters.
-   **`vertex_ai_utils.py`**: Manages all interactions with Google Cloud Vertex AI, including client initialization (handling `PROJECT_ID` and `LOCATION` dynamically or from `config.py`), model loading (as singletons), and core AI functionalities like text generation and embedding.
-   **`helpers/` directory**: Contains utility modules:
    -   **`clothing_utils.py`**: Defines clothing category keywords and functions for text-based clothing type identification.
    -   **`image_utils.py`**: Provides image processing utilities, primarily for resizing.
    -   **`recommender_utils.py`**: Includes helper functions for the recommendation process, like calculating cosine similarity and displaying CLI results.
-   **Core Scripts**:
    -   `embed_wardrobe.py`: Processes wardrobe images to create embeddings.
    -   `recommender.py`: Offers a CLI tool for getting recommendations.
    -   `main.py`: Runs a Flask web application for an interactive experience.

This structure enhances readability, maintainability, and separation of concerns.

## Setup

### 1. Google Cloud Configuration
-   Enable necessary Google Cloud APIs:
    ```bash
    gcloud services enable \
      cloudresourcemanager.googleapis.com \
      aiplatform.googleapis.com
    ```
-   Authenticate with Google Cloud:
    ```bash
    gcloud auth application-default login
    # Set your project ID for quota and billing purposes
    gcloud auth application-default set-quota-project [YOUR_GCP_PROJECT_ID]
    ```
    Alternatively, you can set the `PROJECT_ID` in `config.py` or as an environment variable `MY_PROJECT_ID`. The `LOCATION` for Vertex AI services is also configured in `config.py`.

### 2. Python Dependencies
-   Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Download Sample Images (Optional)
-   Sample wardrobe images can be downloaded using:
    ```bash
    wget https://storage.googleapis.com/public-file-server/genai-downloads/mr-stylist-images.tar.gz
    tar -xzf mr-stylist-images.tar.gz
    ```
    This will create a `static/images/` folder. You can replace these with your own clothing images. This path is configurable in `config.py` (`IMAGE_URI_PATH`).
-   The `people/` folder (if you download the tarball) contains sample images for the CLI recommender.

## Running the Scripts

### 1. Create Vector Embeddings for Your Wardrobe
The `embed_wardrobe.py` script processes all images in the configured wardrobe image directory (`static/images/` by default), generates textual descriptions and their corresponding vector embeddings, and saves this data into a CSV file (`mywardrobe.csv` by default). This CSV acts as the knowledge base for your wardrobe.

-   **To run:**
    ```bash
    python embed_wardrobe.py
    ```
-   **Configuration**:
    -   The input image path (`IMAGE_URI_PATH`) and output CSV file (`WARDROBE_CSV_FILE`) can be modified in `config.py`.
    -   The script uses models and parameters defined in `config.py` and managed by `vertex_ai_utils.py`.

You can also download pre-generated wardrobe embeddings (ensure the model name in the CSV matches the `EMBEDDING_MODEL` in `config.py` for best results):
```bash
# Example for text-embedding-005 (current default)
wget https://storage.googleapis.com/public-file-server/genai-downloads/mywardrobe_2-0-flash_768.csv -O mywardrobe.csv
```

### 2. Get Recommendations via CLI (`recommender.py`)
The `recommender.py` script provides a command-line interface to get outfit recommendations. It takes an image of a person or a look you want to replicate and suggests items from your processed wardrobe.

-   **To run:**
    ```bash
    python recommender.py <path_to_your_input_image> [number_of_recommendations_per_item]
    ```
    -   Example: `python recommender.py people/model_10.JPG 2` (This would return the top 2 matching items from your wardrobe for each piece of clothing identified in `model_10.JPG`).
-   **Note**: For best results, use input images showing the full body, as partially visible items (e.g., cropped pants) might be misinterpreted.

### 3. Run the Flask Web Application (`main.py`)
The `main.py` script launches a Flask web server, providing an interactive UI to upload an image and receive recommendations.

-   **To run:**
    ```bash
    python main.py
    ```
-   The application will typically be available at **`http://0.0.0.0:8080`**.
-   Upload an image of a look you're interested in, and the app will display recommended items from your wardrobe.
-   The upload folder (`UPLOAD_FOLDER`) and allowed file extensions (`ALLOWED_EXTENSIONS`) are configured in `config.py`.

## Future Development (TODO)
-   Fine-tuning of prompts and model parameters for improved accuracy.
-   Incorporate image vector similarity (e.g., using Vertex AI Vision multimodal embeddings directly for image-to-image search) in addition to text-based similarity for potentially more nuanced results.
-   Transition from CSV-based wardrobe storage to a dedicated vector database (e.g., Vertex AI Vector Search) for scalability and more efficient querying, especially for larger wardrobes.
-   Deploy the application components (Flask app, embedding generation) to Google Cloud resources like Cloud Run, App Engine, or GKE.

This refactoring provides a solid foundation for these future enhancements.
