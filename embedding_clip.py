import time
import torch
import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from options.options import Options
import util.dataloader as dl
from sklearn.preprocessing import robust_scale
from sklearn.manifold import locally_linear_embedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import k_means, KMeans

import clip

def generate_embeddings_clip(reencode = False, normalize_cols = True, estimate_k = True, fix_labels = True, do_embedding = True):
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
            
            # load text before the loop
            text = clip.tokenize(opt.clip_categories).to(device)
            print(f"Finding categories {opt.clip_categories} using CLIP")
            
            # sleeping for tqdm
            time.sleep(0.2)
            pbar = tqdm(total = len(datas))
            
            for data, path in datas:
                
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
                
            
            # dictionary of encodings being converted to pandas dict
            encodings = pd.DataFrame(data = encodings).T
            encodings.columns = opt.clip_categories
            encodings.to_csv("data/encodings.csv")
            pbar.close()
    else:
        print("Loading encodings...")
        encodings = pd.read_csv("data/encodings.csv", index_col = 0)
        
    print("Encodings retrieved")
    # but we still need them as numpy array for sklearn functions
    encodings_np = encodings.to_numpy()
    
    if normalize_cols:
        encodings_np = (encodings_np - encodings_np.min(axis = 0)) / (encodings_np.max(axis = 0) - encodings_np.min(axis = 0))
    
        # k-means clustering of the data
    if estimate_k:
        kmeans = get_best_kmeans(encodings_np)
        labels = kmeans.labels_
    else:
        centroids, labels, inertia = k_means(encodings_np, n_clusters = opt.n_clusters)
    
    # adding top 1 label and k-means labels
    encodings["top_1_label"] = encodings[opt.clip_categories].idxmax(axis = 1)
    encodings["labels"] = labels
    
    if fix_labels:
        # renaming labels using CLIP categories for clarity
        encodings = relabel(encodings, opt.clip_categories)
            
    embeddings = encodings.reset_index()
    embeddings.columns = ["path"] + list(embeddings.columns[1:len(embeddings.columns)])
    
    if do_embedding:
        # embedding the encoding vectors, note n_components must be 3 for 3D
        start_embed = time.time()
        print("Embedding Encodings")
        embeds = manifold_function(encodings_np, opt)
        
        print(f"Embedding took {time.time() - start_embed} seconds")
        
        #making a dataframe of embeddings
        embeddings[["embeddings_x", "embeddings_y", "embeddings_z"]] = embeds
    
        
        embeddings = embeddings[[embeddings.columns[0], "embeddings_x", "embeddings_y", "embeddings_z", "labels", "top_1_label"]]
        embeddings.columns = ["path", "embeddings_x", "embeddings_y", "embeddings_z", "labels", "top_1_label"]
        
    #joining embedding data to old data
    datas.dataset.join_new_col(embeddings)
        
    datas.dataset.data.to_csv("data/embeddings.csv", index = False)
    
    return datas

def get_best_kmeans(np_data, k_max = 25, k_min = 2):
    """
    Function to find the best cluster size, based on silhouette score, for 
    data in a numpy array
    ----------
    np_data: numpy array to cluster on
    k_max: default 25, max number of cluseters to test to
    k_min: default 2, cluster size to start testing at
    """
    
    k_dict = {}
    model_dict = {}
    print(f"\tEstimating best k for clustering\n\t\tRange of k: ({k_min}, {k_max})")
    for ks in range(k_min, k_max + 1):
        print(ks)
        kmeans = KMeans(n_clusters = ks).fit(np_data)
        labels = kmeans.labels_    
        
        model_dict[ks] = kmeans
        k_dict[ks] = silhouette_score(np_data, labels)
    best_k = max(k_dict, key = k_dict.get)
    print(f"\t\tBest k-estimated to be {best_k}, with silhouette score of {k_dict[best_k]}")
    return model_dict[best_k]
 
def relabel(data, label_cols, labels_col = "labels"):
    """
    Sets label for generic k-means group label to the closest in the CLIP labels.
    """
    label_pairs = {}
    available_labels = label_cols
    for label in data[labels_col].unique():
        data_label_x = data.loc[data[labels_col] == label, available_labels]
        new_label = data_label_x.sum(axis = 0).idxmax()
        label_pairs[label] = new_label
    data[labels_col] = data[labels_col].replace(label_pairs)
    return data
    
        
def manifold_function(data, opt):
    """
    Wrapper for the manifold reduction function
    ----------
    data: data to be embedded
    opt: options data, this is where the method is
    """
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