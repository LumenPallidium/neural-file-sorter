import os
import pandas as pd
import shutil
from time import sleep

from tqdm import tqdm
from options.options import Options
from embedding import generate_embeddings

def copy_to_new_loc(keep_old_structure = False, 
                    reorganize_col = "labels",
                    rename = True):
    """
    Function that sorts data in the hard-drive based on the labels generated in
    the embedding method. By default, it drops any old subfolders.
    ----------
    keep_old_structure: default false, boolean that controls if old folder
        names will be replicated within new directory structure
    reogranize_col: the column to use for reorganizing. defaults to k-means 
        label.
    rename: default true, boolean that controls if images will be renamed to
        include their label. Requires that the data was clustered hierarchically.
    """
    assert not (rename and keep_old_structure), "rename and keep_old_structure cannot both be true"
    opt = Options()

    if not os.path.exists("data/embeddings.csv"):
        generate_embeddings()
    df = pd.read_csv("data/embeddings.csv")
    
    # the folder where you are copying
    out_folder = opt.out_filepath
    # make the new folder for auto-sorted images
    os.makedirs(out_folder, exist_ok = True)
        
    original_folder = opt.filepath
    print("Copying files...")
    sleep(1) # sleep for tqdm
    for ims, label in tqdm(df[["path", reorganize_col]].itertuples(index = False)):

        if keep_old_structure:
            # when keeping old structure, simply trim off 
            ims_out = ims.replace(original_folder, "")
            folders = ims_out.split("/")
            # removing last item ie the image filename
            folders.pop()
            
            for folder in folders:
                path_str = path_str + folder + "/"
                os.makedirs(path_str, exist_ok = True)
            
        else:
            # when not keeping old structure, extract filename only
            ims_out = ims.split("/")[-1]

            if rename:
                ims_suffix = ims_out.split(".")[-1]
                # rename to include label
                ims_out = str(label) + "." + ims_suffix
                new_path = out_folder +  "/" + ims_out
            else:
                new_path = out_folder + "/" + str(label) + "/" + ims_out
        try:
            # manually adding forward slash cause windows uses \ as os.sep
            shutil.copyfile(ims, new_path)
        except FileNotFoundError:
            print(f"{ims} does not exist")

if __name__ == "__main__":
    copy_to_new_loc()

