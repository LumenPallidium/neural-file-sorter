import time
import torch

from options.options import Options
import util.dataloader as dl
from models.visual_models import *
import models.audio_models


def train(opt):
    """
    Function to train an autoencoder, based on the options. Options can be 
    configured based on yaml file. Model is saved at each epoch with a single 
    name (i.e. only one checkpoint file is generated).
    Output is model saved in checkpoint folder as well as returned by the
    function.
    ----------
    opts: instance of Options class
    """
    datas = dl.DataLoader(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start = time.time()
    
    total_iters = 0
    
    if datas.dataset.mode == "image":
        # current model format, hardcocded, will generalize later
        model = VisAutoEncoder(opt.visual_aa,
                               **opt.visual_aa_args)
        model = model.to(device)
    else:
        raise(AttributeError(f"{datas.dataset.mode} mode is not supported"))
        
    # currently defaulting to adam, will add more optimizer options later
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # currently defaulting to mse loss, will add more loss function options later
    loss_f = torch.nn.MSELoss()
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    for epoch in range(1, opt.epoch_num + 1):
        print(f"Epoch {epoch}")
        epoch_start = time.time()
        
        iter_count = 0
        epoch_loss = 0        

        for i, data in enumerate(datas):
            if iter_count % 500 == 0:
                print(f"\tIteration {iter_count}")
            
            data = data.to(device)

            # undoing accumulation of grad
            optimizer.zero_grad()

            batch_size = opt.batch_size
            total_iters += batch_size
            iter_count += batch_size
            
            # run data through model
            output = model(data)
            
            # calculate loss
            loss = loss_f(output, data)
            
            # do backprop
            loss.backward()
            
            # adjust weights with optimizer based on backprop
            optimizer.step()
            
            epoch_loss += loss.item()

        
        model.save()
        epoch_end = time.time()

        print(f"End of epoch {epoch} \n\tEpoch Loss : {epoch_loss} \n\tEpoch Time : {epoch_end - epoch_start}")
    return model
        