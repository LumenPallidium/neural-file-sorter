import torch
import clip
import imageio
import os
import torchvision.transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
video_path = "mp4.mp4"
output_path = "to_crop/"
name_prefix = "kikiw"



#model, preprocess = clip.load("ViT-B/32", device=DEVICE)

def extract_video_frames(video_path : str, 
                         output_path : str, 
                         name_prefix : str, 
                         name : int = 1, 
                         n_pads = 5, 
                         frame_interval : float = 0.1, 
                         clip_model = None, 
                         preprocess = None, 
                         clip_text : list = ["landscape", "text", "a drawing of a human", "a drawing of a room"],
                         device = DEVICE):
    """
    Function to extract frames from a video and save in a specified folder. 
    ----------
    video_path: path to the video file
    output_path: folder path to save images in
    name_prefix: prefix to give as an image name, eg if name_prefix = "image_"
        images will be saved as "image_x.jpg", where x is a number (see name)
    name: the starting number for saved filenames, defaults to 1 so first image
        downloaded is "{name_prefix}_1", useful to change if you want to
        continue scraping after you have stopped or are continuing from another
        image list
    n_pads : 
    frame_interval : time in seconds to save each frame at, defaults to 0.5s
        
    """
    
    reader = imageio.get_reader(video_path)

    meta_data = reader.get_meta_data()

    fps = meta_data["fps"]
    # get the frames divisible by number
    frames_to_grab = int(frame_interval * fps)
    
    total_frames = int(fps * meta_data["duration"])
    
    # text doesn't vary so tokenize outside the loop
    text = clip.tokenize(clip_text).to(device)
    
    # count by frames to grab
    for frame_number in tqdm(range(0, total_frames, frames_to_grab)):
        image = reader.get_data(frame_number)
        # converts eg number 11 to 00011, makes sorting images better
        number = str(name)
        number = "0" * (n_pads - len(number)) + number
        if clip_model is None:
            imageio.imwrite(output_path + name_prefix + number + ".png", image)
            name += 1
        else:
            with torch.no_grad():
                image_copy = image
                # image arrives in H, W, C form so we move axis[2] to axis[0] i.e. into C, H, W form
                image = Image.fromarray(np.moveaxis(image, 2, 0), 'RGB')
                image = preprocess(image).unsqueeze(0)
                logits_per_image, logits_per_text = clip_model(image, text)
                
                mean_activation = logits_per_image.softmax(dim=-1).cpu().numpy()
                
                if mean_activation[0][0] > 0.3:
                    imageio.imwrite(output_path + name_prefix + number + ".png", image_copy)
                    name += 1
            
extract_video_frames(video_path, output_path, name_prefix, clip_model = None, preprocess = None)