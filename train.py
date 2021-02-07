import time
import torch

from options.options import Options
import util.dataloader as dl
from models.visual_models import *
import models.audio_models


if __name__ == '__main__':
    opt = Options()
    data = dl.DataLoader(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start = time.time()
    
    total_iters = 0
    
    if data.dataset.mode == "image":
        #current model format, hardcocded, will generalize later
        model = VisAutoEncoder([(3, 128, 7, 1, 4), (128, 256, 3, 2, 2), (256, 512, 3, 2, 2)])
        model = model.to(device)
    else:
        raise(AttributeError(f"{data.dataset.mode} mode is not supported"))
        
    #currently defaulting to adam, will add more optimizer options later
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    #currently defaulting to mse loss, will add more loss function options later
    loss_f = torch.nn.MSELoss()
        
        
    for epoch in range(0, opt.epoch_num):
        print(f"Epoch {epoch}")
        epoch_start = time.time()
        
        iter_count = 0
        epoch_loss = 0        

        for i, data in enumerate(data):
            if iter_count % 500 == 0:
                print(f"Iteration {iter_count}")
            
            data = data.to(device)

            #undoing accumulation of grad
            optimizer.zero_grad()

            batch_size = opt.batch_size
            total_iters += batch_size
            iter_count += batch_size
            
            #run data through model
            output = model(data)
            
            #calculate loss
            loss = loss_f(output, data)
            
            #do backprop
            loss.backward()
            
            #adjust weights with optimizer based on backprop
            optimizer.step()
            
            epoch_loss += loss.item()

        
        model.save()
        epoch_end = time.time()

        print(f"End of epoch {epoch} \nEpoch Loss : {epoch_loss} \nEpoch Time : {epoch_end - epoch_start}")