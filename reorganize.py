import os
import pandas as pd
import shutil
from time import sleep

from tqdm import tqdm
from options.options import Options

def copy_to_new_loc(keep_old_structure = False, new_folder = "auto_sorted/", reorganize_col = "labels"):
    """
    Function that sorts data in the hard-drive based on the labels generated in
    the embedding method. By default, it drops any old subfolders.
    ----------
    keep_old_structure: default false, boolean that controls if old folder
        names will be replicated within new directory structure
    new_folder: options data, this is where the method is
    reogranize_col: the column to use for reorganizing. defaults to k-means 
        label.
    """
    opt = Options()

    df = pd.read_csv("data/embeddings.csv", index_col = 0)
    
    # the folder where you are copying
    out_folder = opt.out_filepath
    # make the new folder for auto-sorted images
    try:
        os.mkdir(out_folder + new_folder)
    except FileExistsError:
        print(f"{new_folder} already exists, no need for new folder")
        
    original_folder = opt.filepath
    sleep(1)
    for ims, label in tqdm(df[["path", reorganize_col]].itertuples(index = False)):
        path_str = out_folder + new_folder + str(label) + "/"
        # start by making a folder for the label
        try:
            os.mkdir(path_str)
        except FileExistsError:
            pass
        
        if keep_old_structure:
            # when keeping old structure, simply trim off 
            ims_out = ims.replace(original_folder, "")
            folders = ims_out.split("/")
            # removing last item ie the image filename
            folders.pop()
            
            for folder in folders:
                path_str = path_str + folder + "/"
                
                try:
                    os.mkdir(path_str)
                except FileExistsError:
                    pass
            
        else:
            # when not keeping old structure, extract filename only
            ims_out = ims.split("/")[-1]
        try:
            # manually adding forward slash cause windows uses \ as os.sep
            shutil.copyfile(ims, out_folder + new_folder + str(label) + "/" + ims_out)
        except FileNotFoundError:
            print(f"{ims} does not exist")

