import torch
import clip
from PIL import Image
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

detokenizer = SimpleTokenizer()

model, preprocess = clip.load("ViT-B/32", device=device)

def generate_encoding_from_tensor(in_tensor, out_size = 77, sot_token = 49406, eot_token = 49407, device = device):
    sot = torch.tensor([sot_token]).to(device)
    eot = torch.tensor([eot_token]).to(device)
    
    if in_tensor.dim() == 1:
        out_tensor = torch.cat((sot, in_tensor, eot))
        pad = out_size - out_tensor.shape[1]
        out_tensor = torch.cat((out_tensor, torch.zeros(pad)))
    
    elif in_tensor.dim() == 2:
        sot = sot.repeat(in_tensor.shape[0], 1)
        eot = eot.repeat(in_tensor.shape[0], 1)
        
        
        out_tensor = torch.cat((sot, in_tensor, eot), dim = 1)
        pad = out_size - out_tensor.shape[1]
        out_tensor = torch.cat((out_tensor, torch.zeros((in_tensor.shape[0], pad), device = device)), dim = 1)
    
    else:
        raise ValueError(f"Input tensor must be dimension 1 or 2, not {in_tensor.dim()}")
    
    return out_tensor
    


def project_to_text(image, clip_model, n_iterations = 1000, n_tokens = 5, token_set_size = 49405, device = device):
    text_encoding = (torch.rand(1, n_tokens) * token_set_size).to(device)
    
    text_encoding.requires_grad = True
    clip_model.requires_grad = False
    
    image_encoding = clip_model.encode_image(image)
    image_encoding = image_encoding / image_encoding.norm(dim=-1, keepdim=True)
    
    optimizer = torch.optim.AdamW([text_encoding])
    
    for step in tqdm(range(n_iterations)):
        
        text_encoding_round = generate_encoding_from_tensor(text_encoding)
        text_encoding_round = text_encoding_round.long()
        
        text_features = clip_model.encode_text(text_encoding_round)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        loss = -1 * (image_encoding @ text_features.t()).abs().sum()
        print(text_encoding)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()
        
    return text_encoding