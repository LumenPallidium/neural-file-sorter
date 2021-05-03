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
from sklearn.preprocessing import robust_scale
from sklearn.manifold import locally_linear_embedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import k_means


def generate_embeddings(retrain = False, reencode = False, quick = True, normalize_cols = False):
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
    
    def decode_images(centroid_list, model, device):
        print("Converting centroids to images...")
        image_list = {}
        for i, encoded_image in enumerate(centroid_list):
            encoded_image = torch.Tensor([encoded_image])
            encoded_image = encoded_image.to(device)
            image = model.decoder(encoded_image)
            name = f"label_{i}"
            image_list[name] = image.cpu().detach()
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
    datas = dl.DataLoader(opt, transform_mode = "resize only", return_path = True)
    
    # following loop generates the encodings, simply passing data through
    # the encoder and condensing, see encode_image function
    encodings = {}
    
    if reencode or not os.path.exists("data/encodings.csv"):
        start_encode = time.time()
        print("Generating Encodings")
        
        # sleeping for tqdm
        time.sleep(0.2)
        pbar = tqdm(total = len(datas))
        
        for data, path in datas:
            

            pbar.update(1)
            
            data = data.to(device)
    
            encoding= model.encode_image(data)
            
            encodings[path] = encoding
            
        pbar.close()
        
        # dictionary of encodings being converted to pandas dict
        encodings = pd.DataFrame(data = encodings).T
        print(f"Encodings of length {encoding.shape}")
        encodings.to_csv("data/encodings.csv")
        print(f"Encoding took {time.time() - start_encode}")
    else:
        print("Loading encodings...")
        encodings = pd.read_csv("data/encodings.csv", index_col = 0)
        
    print("Encodings retrieved")    
    # but we still need them as numpy array for sklearn functions
    encodings_np = encodings.to_numpy()
    
    if normalize_cols:
        encodings_np = (encodings_np - encodings_np.min(axis = 0)) / (encodings_np.max(axis = 0) - encodings_np.min(axis = 0))
    
    # embedding the encoding vectors, note n_components must be 3 for 3D
    start_embed = time.time()
    print("Embedding Encodings")
    encodings_np = robust_scale(encodings_np)
    embeds = manifold_function(encodings_np, opt)
    
    # k-means clustering of the data
    centroids, labels, inertia = k_means(encodings_np, n_clusters = opt.n_clusters)
    
    print(f"Embedding took {time.time() - start_embed} seconds")
    if not quick:
        label_images = decode_images(centroids, model, device)
    else:
        label_images = None
    
    #making a dataframe of embeddings
    encodings[["embeddings_x", "embeddings_y", "embeddings_z"]] = embeds
    embeddings = encodings.reset_index()
    embeddings = embeddings[[embeddings.columns[0], "embeddings_x", "embeddings_y", "embeddings_z"]]
    embeddings["labels"] = labels
    embeddings.columns = ["path", "embeddings_x", "embeddings_y", "embeddings_z", "labels"]
    
    #joining embedding data to old data
    datas.dataset.join_new_col(embeddings)
    
    datas.dataset.data.to_csv("data/embeddings.csv", index = False)
    
    return datas, label_images
    
        
def manifold_function(data, opt):
    method = opt.embedding_method.lower()
    if method == "lle":
        embeds, err = locally_linear_embedding(data, **opt.general_manifold_params)
    elif method == "mds":
        embeds = MDS(**opt.general_manifold_params).fit_transform(data)
    elif method == "isomap":
        embeds = Isomap(**opt.general_manifold_params).fit_transform(data)
    elif method == "t-sne":
        embeds = TSNE(**opt.general_manifold_params).fit_transform(data)
    elif method == "pca":
        embeds = PCA(**opt.general_manifold_params).fit_transform(data)
    else:
        raise ValueError(f"Embedding method {method} not valid. Valid methods are: lle, mds, isomap, t-sne, pca")
    return embeds
    

    