import os
import pandas as pd
import mimetypes
import torch.utils.data
import torchvision.transforms
mimetypes.init()
from PIL import Image
import av

def map_dirs(filepath,
             supported = [".png", ".jpg", ".jpeg", ".bmp"],
             video_supported = [".mp4", ".avi", ".webm", ".mov", ".m4v"],
             depth = 0):
    """
    Given a filepath, this function builds a map of the directory, returning it
    as a pandas dataframe with features like filetypes
    ----------
    filepath: string of the file location
    """
    files = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(filepath):
        if count > depth:
            break
        for file in filenames:
            abs_path = os.path.join(os.path.abspath(dirpath), file)
            filename, extension = os.path.splitext(file)
            filetype = mimetypes.guess_type(file)[0]
            try:
                filetype = filetype.split('/')[0]
            except AttributeError:
                filetype = "Other"
            files.append([abs_path, filename, extension, filetype])
        count += 1
    df = pd.DataFrame(files, columns=["path", "filename", "extension", "filetype"])

    if os.name == "nt":
        #replace backslashes with forward in windows
        df["path"] = df["path"].str.replace("\\", "/")

    if video_supported:
        # if video support is provided, call it "image"
        df.loc[df["filetype"] == "video", "filetype"] = "image"
    
    #adding flag for supported file types
    df["support"] = "Unsupported"
    df.loc[df["extension"].str.lower().isin(supported + video_supported), "support"] = "Supported"
    
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


def transform_im(pil_im, out_size = (256, 256), transform_mode = "affine and scale", rot = (-10, 10), trans = (0.2, 0.2), rand_scale = (0.2, 1)):
    """
    Transforms a PIL Image. 
    ----------
    pil_im: a PIL image file. Note pytorch tensors would also work
    """
    if transform_mode == "affine and scale":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=rot, translate=trans, interpolation=Image.BILINEAR),
            torchvision.transforms.RandomResizedCrop(size = out_size, scale = rand_scale),
            torchvision.transforms.ToTensor()])
    elif transform_mode == "resize only":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(out_size[0]),
            torchvision.transforms.CenterCrop(out_size),
            torchvision.transforms.ToTensor()])
    elif transform_mode == "clip_mode":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=Image.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
    else:
        raise(ValueError(f"{transform_mode} transform mode is invalid"))
    return transforms(pil_im)

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for interacting with the Pytorch dataloader
    ----------
    filepath: the directory to be mapped, a string
    mode: used to control whether filetypes are images or audio, string
    transform: function which transforms the input
    """
    def __init__(self, filepath, mode = "image", transform = transform_im, transform_mode = "affine and scale", out_size = (256, 256), return_path = False):
        
        data = map_dirs(filepath)

        self.filepath = filepath
        self.data_original = data
        self.summary = summarize_filetypes(self.data_original)
        self.transform = transform
        self.transform_mode = transform_mode
        self.out_size = out_size
        self.mode = mode
        self.return_path = return_path
        #need to initialize to drop unsupported filetypes
        self.initialize()

    def initialize(self):
        #needs to be done to filter unsupported filetypes
        data = self.data_original.copy()
        self.data = data[(data["support"] == "Supported") & (data["filetype"] == self.mode)].reset_index()
        print(f"Data initialized: Unsupported filetypes and non-{self.mode} filetypes dropped")
        
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
            img = self.transform(img, out_size = self.out_size, transform_mode = self.transform_mode)
        if self.return_path:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.data)
    
    #wrapper for image loading
    def load_image(self, im_path):
        if im_path.split(".")[-1] in ["mp4", "avi", "webm", "mov", "m4v"]:
            stream = av.open(im_path)
            for frame in stream.decode(video = 0):
                frame = frame.to_ndarray(format = "rgb24")
                break
            stream.close()
            im = Image.fromarray(frame)
        else:
            im = Image.open(im_path).convert('RGB')
        return im
    
    def mode_update(self, mode_str):
        """
        In case you want to update the mode of the dataset after creating it
        ----------
        mode_str: used to control whether filetypes are images or audio
        """
        self.mode = mode_str
        return self
        
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
        self.dataset = Dataset(opt.filepath, out_size = opt.visual_aa_args["in_size"], **kwargs)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

