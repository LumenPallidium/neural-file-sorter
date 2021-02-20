import time
import torch
import pandas as pd
import os

from tqdm import tqdm
from options.options import Options
import util.dataloader as dl
from models.visual_models import *
import models.audio_models
from train import train
from sklearn.manifold import locally_linear_embedding
from sklearn.cluster import k_means


def generate_embeddings(retrain = False, reencode = False, quick = True):
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
    reencode: default False, if True it regenerates image encodings, even if they already
            exist
    quick: Collapses encoding vectors, default True. If False, a reshaped version of
            the encoding array which can be very large, and will take large amounts of
            space to store and run slowly in embedding and clustering. On the other hand,
            you can generate representative image for the clusters.
    """
    def encode_image(model, image, quick = quick, reduced_size = 512):
        encoding = model.encoder(image)
        encoding_vector = encoding.detach()
        if quick:
            # takes a long time with default sizes so if quick is enabled
            # rescale to a better length (default close to 512)
            initial_shape = encoding_vector.shape
            initial_length = initial_shape[1]
            # formula used from here, assuming stride = kernel
            # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d
            kernel = int(initial_length / reduced_size)
            pooler = nn.AvgPool1d(kernel_size = kernel)
            encoding_vector = pooler(torch.unsqueeze(encoding_vector, 1))
            encoding_vector = torch.squeeze(encoding_vector, 1)
        encoding_vector = encoding_vector[0].cpu()
        return encoding_vector, initial_shape
    
    def decode_images(centroid_list, out_shape, model, device):
        print("Converting centroids to images...")
        image_list = {}
        for i, encoded_image in enumerate(centroid_list):
            encoded_image = encoded_image.reshape(out_shape)
            encoded_image.to(device)
            image = model.decoder(encoded_image)
            name = f"label_{i}"
            image_list[name] = image
        return image_list
    
    opt = Options()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # retrain conditions
    if retrain or not os.path.exists("ckpts/default_model.pt"):
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
    
    if reencode or not os.path.exists("data/encodings.csv"):
        start_encode = time.time()
        print("Generating Encodings")
        
        pbar = tqdm(total = len(datas))
        
        for data, path in tqdm(datas):
            

            pbar.update(1)
            
            data = data.to(device)
    
            encoding, initial_shape = encode_image(model, data)
            
            encodings[path] = encoding
            
        pbar.close()
        
        # dictionary of encodings being converted to pandas dict
        encodings = pd.DataFrame(data = encodings).T
        print(f"Encodings of length {encoding.shape} created from array of shape {initial_shape}")
        encodings.to_csv("data/encodings.csv")
        print(f"Encoding took {time.time() - start_encode}")
    else:
        print("Loading encodings...")
        encodings = pd.read_csv("data/encodings.csv")
        print(encodings.columns)
        
    print("Encodings retrieved")    
    # but we still need them as numpy array for sklearn functions
    encodings_np = encodings.to_numpy()
    
    # embedding the encoding vectors, note n_components must be 3 for 3D
    print("Embedding Encodings")
    embeds, err = locally_linear_embedding(encodings_np, n_neighbors = opt.lll_neighbors, n_components = 3)
    
    # k-means clustering of our data, i'm only interested in labels tho
    centroids, labels, inertia = k_means(encodings_np, n_clusters = opt.n_clusters)
    
    if reencode and not quick:
        label_images = decode_images(centroids, initial_shape, model, device)
    else:
        label_images = None
    
    #making a dataframe of embeddings
    encodings[["embeddings_x", "embeddings_y", "embeddings_z"]] = embeds
    embeddings = encodings.reset_index()[["level_0", "embeddings_x", "embeddings_y", "embeddings_z"]]
    embeddings["labels"] = labels
    embeddings.columns = ["path", "embeddings_x", "embeddings_y", "embeddings_z", "labels"]
    
    #joining embedding data to old data
    datas.dataset.join_new_col(embeddings)
    
    datas.dataset.data.to_csv("data/embeddings.csv", index = False)
    
    return datas, label_images
    
        
        

    