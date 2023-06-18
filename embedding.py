import time
import torch
import pandas as pd
import os
from tqdm import tqdm
from options.options import Options
import util.dataloader as dl
from models.visual_models import *
from models.hierarchical_clusterer import HierarchicalClusterer
import models.audio_models
from train import train
from sklearn.preprocessing import robust_scale
from sklearn.manifold import locally_linear_embedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import k_means, KMeans
from sklearn.metrics import silhouette_score

import clip

def load_model(retrain, opt, device):
    # retrain conditions
    if retrain or not os.path.exists("ckpts/default_model.pt"):
        model = train(opt)
    else:
        model = VarVisAutoEncoder(**opt.visual_aa_args) if opt.variational else VisAutoEncoder(**opt.visual_aa_args)
        model.load_state_dict(torch.load("ckpts/default_model.pt"))
        model = model.to(device)
    return model

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

def setup_opts():
    opt = Options()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #set batch_size to 1 and reload dataset
    #must be 1 or extra data will be ignored later
    opt.batch_size = 1
    transform_mode = "clip_mode" if opt.use_clip else "resize only"
    datas = dl.DataLoader(opt, transform_mode = transform_mode, return_path = True)
    
    # following loop generates the encodings, simply passing data through
    # the encoder and condensing, see encode_image function
    encodings = {}
    return opt, device, datas, encodings

def encode(use_clip, use_clip_labels, model, datas, encodings, opt, device):
    start_encode = time.time()
    print("Generating Encodings")
    
    if use_clip and use_clip_labels:
        # load text before the loop
        text = clip.tokenize(opt.clip_categories).to(device)
        print(f"Finding categories {opt.clip_categories} using CLIP")
    
    with torch.no_grad():
        # sleeping for tqdm
        time.sleep(0.2)
        pbar = tqdm(total = len(datas))
        for data, path in datas:
            
            pbar.update(1)
            
            data = data.to(device)
    
            if use_clip and use_clip_labels:
                logits_per_image, logits_per_text = model(data, text)
                probs = logits_per_image.softmax(dim=-1)[0].cpu().numpy()
                
                encodings[path] = probs
            else:
                encoding = model.encode_image(data)[0].cpu().numpy()
                encodings[path] = encoding
            
        pbar.close()
    
    # dictionary of encodings being converted to pandas dict
    encodings = pd.DataFrame(data = encodings).T.reset_index()
    if use_clip_labels:
        encodings.columns = ["path"] + opt.clip_categories
    else:
        encodings.columns = ["path"] + [f"dim_{i}" for i in range(encodings.shape[1] - 1)]
    encodings.to_csv("data/encodings.csv", index = False)
    print(f"Encoding took {time.time() - start_encode}")
    return encodings

def generate_embeddings():
    """
    Function to generate the embeddings of files. Uses a trained autoencoder
    to first encode the image, then a dimensionality reduction algorithim is 
    run on each encoded image to embed it in 3D space.
    Additionally, clusters are generated with k-means clustering for better
    visualizing the data.
    Output data is saved as a datafile for later plotting, as well as returned
    in a Dataset object.
    """
    
    opt, device, datas, encodings = setup_opts()

    # special args get their own variables
    use_clip = opt.use_clip
    use_clip_labels = opt.use_clip_labels
    retrain = opt.retrain
    reencode = opt.reencode
    quick = opt.quick
    estimate_k = opt.estimate_k
    regenerate_embedding = opt.regenerate_embedding
    fix_labels = opt.fix_labels

    if use_clip:
        model, preprocess = clip.load("ViT-B/32", device=device)
    else:
        model = load_model(retrain, opt, device)

    #disabling training mode (eg removes dropout)
    model.train(mode=False)
    os.makedirs("data", exist_ok = True)
    
    if reencode or not os.path.exists("data/encodings.csv"):
        encodings = encode(use_clip, use_clip_labels, model, datas, encodings, opt, device)
    else:
        print("Loading encodings...")
        encodings = pd.read_csv("data/encodings.csv")
        
    print("Encodings retrieved")    
    # but we still need them as numpy array for sklearn functions
    encodings_np = encodings.drop("path", axis = 1).to_numpy()
    
    encodings_np = robust_scale(encodings_np)

    label_images = None
    # adding top 1 label and k-means labels
    if use_clip and use_clip_labels:
        encodings["labels"] = encodings[opt.clip_categories].idxmax(axis = 1)
        if fix_labels:
            # renaming labels using CLIP categories for clarity
            encodings = relabel(encodings, opt.clip_categories)  

    else:
        if opt.use_hc:
            hc = HierarchicalClusterer()
            labels = hc.label_data(encodings_np)
        else:
            # k-means clustering of the data
            if estimate_k:
                kmeans = get_best_kmeans(encodings_np)
                labels = kmeans.labels_
            else:
                centroids, labels, inertia = k_means(encodings_np, n_clusters = opt.n_clusters)
                if not quick:
                    label_images = decode_images(centroids, model, device)
        encodings["labels"] = labels
    
    if regenerate_embedding or (not os.path.exists("data/embeddings.csv")):
        # embedding the encoding vectors, note n_components must be 3 for 3D
        start_embed = time.time()
        print("Embedding Encodings")
        encodings_np = robust_scale(encodings_np)
        embeds = manifold_function(encodings_np, opt)
        print(f"Embedding took {time.time() - start_embed} seconds")

        #making a dataframe of embeddings
        encodings[["embeddings_x", "embeddings_y", "embeddings_z"]] = embeds
        embeddings = encodings.loc[:, ["path", "embeddings_x", "embeddings_y", "embeddings_z", "labels"]]
        embeddings.to_csv("data/embeddings.csv", index = False)
    else:
        print("Loading embeddings...")
        embeddings = pd.read_csv("data/embeddings.csv")

    #joining embedding data to old data
    datas.dataset.join_new_col(embeddings)

    return datas, label_images

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
    """Wrapper for manifold embedding functions, to allow for easy switching between them
    with a string input"""
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
    
if __name__ == "__main__":
    generate_embeddings()
    