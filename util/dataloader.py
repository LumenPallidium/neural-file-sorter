import os
import pandas as pd
import mimetypes
import torch.utils.data
import torchvision.transforms
mimetypes.init()
from PIL import Image

def map_dirs(filepath):
    """
    Given a filepath, this function builds a map of the directory, returning it
    as a pandas dataframe with features like filetypes
    ----------
    filepath: string of the file location
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(filepath):
        for file in filenames:
            abs_path = os.path.join(os.path.abspath(dirpath), file)
            filename, extension = os.path.splitext(file)
            filetype = mimetypes.guess_type(file)[0]
            try:
                filetype = filetype.split('/')[0]
            except AttributeError:
                filetype = "Other"
            files.append([abs_path, filename, extension, filetype])
    df = pd.DataFrame(files, columns=["path", "filename", "extension", "filetype"])
    if os.name == 'nt':
        #replace backslashes with forward in windows
        df["path"] = df["path"].str.replace("\\", "/")
    
    #adding flag for supported file types
    supported = [".png", ".jpg", ".jpeg", ".bmp"]
    df["support"] = "Unsupported"
    df.loc[df["extension"].str.lower().isin(supported), "support"] = "Supported"
    
    return df

def summarize_filetypes(dir_map):
    """
    Given a directory map dataframe, this returns a simple summary of the
    filetypes contained therein and whether or not those are supported or not
    ----------
    dir_map: pandas dictionary with columns path, extension, filetype, support;
            this is the ouput of the map_dirs function
    """
    out_df = dir_map.groupby(["support", "filetype", "extension"])["path"].count()
    return out_df


def transform_im(pil_im, scale = (256, 256)):
    """
    Simple transform, scales a PIL image to desired size and returns it as a
    pytorch tensor
    ----------
    pil_im: a PIL image file. Note pytorch tensors would also work
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(scale),
        torchvision.transforms.ToTensor()])
    return transforms(pil_im)

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for interacting with the Pytorch dataloader
    ----------
    filepath: the directory to be mapped, a string
    mode: used to control whether filetypes are images or audio, string
    transform: function which transforms the input
    """
    def __init__(self, filepath, mode = "image", transform = transform_im, scale = (256, 256), return_path = False):
        
        data = map_dirs(filepath)

        self.filepath = filepath
        self.data_original = data
        self.summary = summarize_filetypes(self.data_original)
        self.transform = transform
        self.scale = scale
        self.mode = mode
        self.return_path = return_path
        
    #pytorch dataloader likes len and getitem methods in datasets
    def __getitem__(self, index):
        path = self.data.loc[index, "path"]
        if self.mode == "image":
            img = self.load_image(path)
        elif self.mode == "audio":
            ##todo
            raise(AttributeError("Audio not yet supported"))
        else:
            raise(AttributeError(f"{self.mode} filetype is not supported"))
            
        if self.transform is not None:
            img = self.transform(img, scale = self.scale)
        if self.return_path:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.data)
    
    #wrapper for image loading
    def load_image(self, im_path):
        return Image.open(im_path).convert('RGB')
    
    def mode_update(self, mode_str):
        """
        In case you want to update the mode of the dataset after creating it
        ----------
        mode_str: used to control whether filetypes are images or audio
        """
        self.mode = mode_str
        return self
    
    def initialize(self):
        #needs to be done to filter unsupported filetypes
        data = self.data_original.copy()
        self.data = data[(data["support"] == "Supported") & (data["filetype"] == self.mode)].reset_index()
        print(f"Data initialized: Unsupported filetypes and non-{self.mode} filetypes dropped")
        
    def join_new_col(self, new_data, merge_col = "path"):
        out_data = self.data.copy()
        out_data = out_data.merge(new_data, how = "left", on = merge_col)
        self.data = out_data
        

class DataLoader():
    """
    DataLoader class for interacting with the Pytorch dataloader
    ----------
    opt : object of class Options defined in options/options.py
    """
    def __init__(self, opt, **kwargs):
        self.opt = opt
        self.dataset = Dataset(opt.filepath, scale = opt.visual_aa_args["in_size"], **kwargs)
        #need to initialize to drop unsupported filetypes
        self.dataset.initialize()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

