import time
import torch
import pandas as pd
import os

from tqdm import tqdm
from options.options import Options
import util.dataloader as dl
from sklearn.preprocessing import robust_scale
from sklearn.manifold import locally_linear_embedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import k_means

import clip

def generate_embeddings_clip(reencode = False):
    """
    Function to generate the embeddings of files. Uses OpenAI's CLIP
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
    
    opt = Options()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load clip model
    model, preprocess = clip.load("ViT-B/32", device=device)

    
    #set batch_size to 1 and reload dataset
    #must be 1 or extra data will be ignored later
    opt.batch_size = 1
    
    #clip_mode is identical to the preprocess function CLIP creates
    datas = dl.DataLoader(opt, transform_mode = "clip_mode", return_path = True)
    
    # following loop generates the encodings, simply passing data through
    # the encoder and condensing, see encode_image function
    encodings = {}
    
    if reencode or not os.path.exists("data/encodings.csv"):
        with torch.no_grad():
            start_encode = time.time()
            print("Generating Encodings")
            
            # sleeping for tqdm
            time.sleep(0.2)
            pbar = tqdm(total = len(datas))
            
            # load text before the loop
            text = clip.tokenize(opt.clip_categories).to(device)
            
            for data, path in tqdm(datas):
                
                data = data.to(device)
        
                # encoding the images, not much point in reassigning but
                # doing it anyway
                im_encoding = model.encode_image(data)
                text_encoding = model.encode_text(text)
    
                # generating logits for text encodings and converting to
                # probabilities
                logits_per_image, logits_per_text = model(data, text)
                probs = logits_per_image.softmax(dim=-1)[0].cpu().numpy()
                
                encodings[path] = probs
                pbar.update(1)
                
            pbar.close()
            
            # dictionary of encodings being converted to pandas dict
            encodings = pd.DataFrame(data = encodings).T
            encodings.columns = opt.clip_categories
            encodings.to_csv("data/encodings.csv")
    else:
        print("Loading encodings...")
        encodings = pd.read_csv("data/encodings.csv", index_col = 0)
        
    print("Encodings retrieved")    
    # but we still need them as numpy array for sklearn functions
    encodings_np = encodings.to_numpy()
    
    # embedding the encoding vectors, note n_components must be 3 for 3D
    start_embed = time.time()
    print("Embedding Encodings")
    encodings_np = robust_scale(encodings_np)
    embeds = manifold_function(encodings_np, opt)
    
    # k-means clustering of the data
    centroids, labels, inertia = k_means(encodings_np, n_clusters = opt.n_clusters)
    
    print(f"Embedding took {time.time() - start_embed} seconds")
    
    #making a dataframe of embeddings
    encodings[["embeddings_x", "embeddings_y", "embeddings_z"]] = embeds
    embeddings = encodings.reset_index()
    embeddings = embeddings[[embeddings.columns[0], "embeddings_x", "embeddings_y", "embeddings_z"]]
    embeddings["labels"] = labels
    embeddings.columns = ["path", "embeddings_x", "embeddings_y", "embeddings_z", "labels"]
    
    #joining embedding data to old data
    datas.dataset.join_new_col(embeddings)
    
    datas.dataset.data.to_csv("data/embeddings.csv", index = False)
    
    return datas
    
        
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