import time
import torch
import pandas as pd
import os

from options.options import Options
import util.dataloader as dl
from models.visual_models import *
import models.audio_models
from train import train
from sklearn.manifold import locally_linear_embedding
from sklearn.cluster import k_means


def generate_embeddings(retrain = False):
    """
    Function to generate the embeddings of files. Uses a trained autoencoder
    to first encode the image, then a dimensionality reduction algorithim is 
    run on each encoded image to embed it in 3D space.
    Additionally, clusters are generated with k-means clustering for better
    visualizing the data.
    Output data is saved as a datafile for later plotting, as well as returned
    in a Dataset object.
    ----------
    retrain: default False, parameter to retrain neural network if a trained
            one already exists
    """
    def encode_image(model, image):
        encoding = model.encoder(image)
        # not sure if there is a way to un-hardcode this
        # 1st axis is batch (should be just 1 image now), 2nd is out_channels of convolution, 
        # 3rd and 4th are size h,w of convolved image
        # thus, the following line extracts the first batch and sums over the image
        # dimensions
        # thus the dimension of the embedding vector is equal to the channel size
        # of the last convolution layer's out_channels
        encoding_vector = encoding[0].cpu().detach().sum(dim = [1, 2]).numpy()
        return encoding_vector
    
    opt = Options()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # retrain conditions
    if retrain == True or not os.path.exists("ckpts/default_model.pt"):
        model = train(opt)
    else:
        model = VisAutoEncoder(opt.visual_aa,
                                   **opt.visual_aa_args)
        model.load_state_dict(torch.load("ckpts/default_model.pt"))
        model = model.to(device)

    
    #disabling training mode (eg removes dropout)
    model.train(mode=False)

    
    #set batch_size to 1 and reload dataset
    #must be 1 or extra data will be ignored later
    opt.batch_size = 1
    datas = dl.DataLoader(opt, return_path = True)
    
    # following loop generates the encodings, simply passing data through
    # the encoder and condensing, see encode_image function
    encodings = {}
    
    iter_count = 0
    
    print("Generating Encodings")
    for data, path in datas:
        
        if iter_count % 500 == 0:
                print(f"\tIteration {iter_count}")
        iter_count += 1
        
        data = data.to(device)

        encoding = encode_image(model, data)
        
        encodings[path] = encoding
    
    # dictionary of encodings being converted to pandas dict
    encodings = pd.DataFrame(data = encodings).T
    
    # but we still need them as numpy array for sklearn functions
    encodings_np = encodings.to_numpy()
    
    # embedding the encoding vectors, note n_components must be 3 for 3D
    print("Embedding Encodings")
    embeds, err = locally_linear_embedding(encodings_np, n_neighbors = opt.lll_neighbors, n_components = 3)
    
    # k-means clustering of our data, i'm only interested in labels tho
    centroids, labels, inertia = k_means(encodings_np, n_clusters = opt.n_clusters)
    
    #making a dataframe of embeddings
    encodings[["embeddings_x", "embeddings_y", "embeddings_z"]] = embeds
    embeddings = encodings.reset_index()[["level_0", "embeddings_x", "embeddings_y", "embeddings_z"]]
    embeddings["labels"] = labels
    embeddings.columns = ["path", "embeddings_x", "embeddings_y", "embeddings_z", "labels"]
    
    #joining embedding data to old data
    datas.dataset.join_new_col(embeddings)
    
    datas.dataset.data.to_csv("data/embeddings.csv", index = False)
    
    return datas
    
        
        

    