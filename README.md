# MR Stylist
[M]ultimodal [R]AG Stylist

- download images:
```
wget https://storage.googleapis.com/public-file-server/genai-downloads/mr-stylist-images.tar.gz
```

**NOTE:** the `images/` folder contains images from some of my clothing, but please feel free to add more of (or replace with) your own 

**NOTE 2:** the `people/` folder contains images of celebrities/clothing models, but again, please feel free to Google search your own



## Create Vector Embeddings for Wardrobe
`embed_wardrobe.py` will create vector embeddings of each piece of clothing in the `images/` folder and write it to a CSV file

`recommender.py` takes 2 arguments, the first is an image of a model/celeb/influencer you would like to replicate the "look" of. The second is the number of results you would like to see (i.e. `1` for top result)


## TODO
A lot...
