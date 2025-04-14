# MR Stylist
[**M**]ultimodal [**R**]AG **Stylist** is my (prototype) clothing recommender built using Gemini's multimodal capabilites and Retrieval-Augmented Generation (RAG) to reduce halucination.

This prototype can easily be adapted for furniture/interior designs and perhaps groceries to provide alternatives for recipes.

`v0.3.0` is being submitted for the **Devpost/Google AI Hackathon**.  I've posted a [demo on YouTube](https://www.youtube.com/watch?v=g3WcuO87FUI&ab_channel=GlenYu).


## Setup
- enable Google Cloud APIs:
```
gcloud services enable \
  cloudresourcemanager.googleapis.com \
  aiplatform.googleapis.com
```

- authenticate
```
gcloud auth application-default login
gcloud auth application-default set-quota-project [YOUR_GCP_PROJECT_ID]
```


- install Python dependencies:
```
pip install -r requirements.txt
```

- download images:
```
wget https://storage.googleapis.com/public-file-server/genai-downloads/mr-stylist-images.tar.gz
```

**NOTE:** the `static/images/` folder contains images from some of my clothing, but please feel free to add more of (or replace with) your own.  It also is the path used by the index.html template 

**NOTE 2:** the `people/` folder contains images of celebrities/clothing models, but again, please feel free to Google search your own


### Create Vector Embeddings for Wardrobe 
`embed_wardrobe.py` will create vector embeddings of each piece of clothing in the `static/images/` folder and write it to a CSV file. This is inefficient and for prototyping/local testing purposes only. Future state will include a vector database deployed in the cloud (GCP).

You can optionally download the wardrobe embeddings I created here:
```
wget https://storage.googleapis.com/public-file-server/genai-downloads/mywardrobe_1-0-pro-vision.csv

wget https://storage.googleapis.com/public-file-server/genai-downloads/mywardrobe_1-5-pro.csv

wget https://storage.googleapis.com/public-file-server/genai-downloads/mywardrobe_2-0-flash.csv
```

The Gemini 1.0 Pro Vision version is currently being used by `main.py` but you can also test the Gemini 1.5 Pro version (but you will have to also make adjustments to the script)


### How to run
There are 2 ways to run MR Stylist:

Running `main.py` will start a Flask front end which you can reach at **http://localhost:80** where you can upload an image of the look you are trying to replicate.  This will return the top result based on the wardrobe you provided. 

The second method is to run`recommender.py`, which takes 2 arguments, the first is an image of a model/celeb/influencer you would like to replicate the "look" of. The second is the number of results you would like to see (i.e. `1` for top result)

- i.e. `python recommender.py people/model_10.JPG 2`  to return top 2 results for each article of clothing it detected in the submitted photo.


**NOTE:** for best results, try to use models which show the full body (head to toe) as pants that are cropped out in the photo have a high chance of being interpretted as shorts.


## TODO
- tuning / improve accuracy 
- combine with image vector embedding similarity results to produce more accurate results
- deploy to GCP resources
- use a vector database
