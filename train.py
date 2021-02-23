import time
import torch

from tqdm import tqdm
from options.options import Options
import util.dataloader as dl
from models.visual_models import *
import models.audio_models
import numpy as np
import warnings

def l_n_reg(model, device, norm = 2):
    """
    Wrapper to add a weight regularizer, with adjustable norm.
    ----------
    model: model with the weights to be regularized
    device: either cpu or gpu
    norm: default 2, the norm to use in regularization
    """
    reg_loss = torch.tensor(0., requires_grad=True).to(device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, norm)
    return reg_loss


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
    
    def create_loss(opt):
        """
        Somewhat ugly but this works in a more efficient way than having a loss 
        function with many conditionals in it.
        Workings are simple: return a loss function based on model details.
        ----------
        output: output from the neural network, which is a tuple in the VAE case
        data: the input that went into the neural network
        curr_iter: current overall iteration of the model.
        model: the neural network
        opt: options, i.e. an instance of the Options class
        device: device to use
        """
        if opt.variational:
            if opt.regularize:
                def loss_f(output, data, curr_iter, model, opt, device):
                    # for a variational model, use KL Divergence for latent
                    # probability distribution loss, and MSE for reconstruction loss
                    loss_mse = torch.nn.MSELoss()
                    
                    # output of forward in this case is model input, mean/variance hidden layers
                    # the encoded state as a sample, and the ouput state
                    inp = output[0]
                    mean = output[1]
                    log_variance = output[2]
                    encod = output[3]
                    output = output[4]
            
                    # VAE is often unstable initially
                    if curr_iter > 2000:
                        # KL Div loss when goal distribution is assumed standard normal
                        # and posterior approximation normal (not neccesarily standard)
                        # https://arxiv.org/abs/1312.6114
                        kl_loss = opt.beta * (-0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()))
                    else:
                        kl_loss = 0
                    reg_loss = l_n_reg(model, device, norm = opt.reg_norm)
                    loss = kl_loss + opt.reg_penalty * reg_loss + loss_mse(inp, output)
                    return loss
            else:
                def loss_f(output, data, curr_iter, model, opt, device):
                    # for a variational model, use KL Divergence for latent
                    # probability distribution loss, and MSE for reconstruction loss
                    loss_mse = torch.nn.MSELoss()
                    
                    # output of forward in this case is model input, mean/variance hidden layers
                    # the encoded state as a sample, and the ouput state
                    inp = output[0]
                    mean = output[1]
                    log_variance = output[2]
                    encod = output[3]
                    output = output[4]
            
                    # VAE is often unstable initially
                    if curr_iter > 2000:
                        # KL Div loss when goal distribution is assumed standard normal
                        # and posterior approximation normal (not neccesarily standard)
                        # https://arxiv.org/abs/1312.6114
                        kl_loss = opt.beta * (-0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()))
                    else:
                        kl_loss = 0
                    reg_loss = l_n_reg(model, device, norm = opt.reg_norm)
                    loss = kl_loss + opt.reg_penalty * reg_loss + loss_mse(inp, output)
                    return loss
        else:
            if opt.regularize:
                def loss_f(output, data, curr_iter, model, opt, device):
                    # currently defaulting to mse loss, will add more loss function options later
                    loss_f = torch.nn.MSELoss()
                
                    reg_loss = l_n_reg(model, device, norm = opt.reg_norm)
                    loss = loss_f(output, data) + opt.reg_penalty * reg_loss
                    return loss
            else:
                def loss_f(output, data, curr_iter, model, opt, device):
                    # currently defaulting to mse loss, will add more loss function options later
                    loss_f = torch.nn.MSELoss()
                    loss = loss_f(output, data)
                    return loss
        return loss_f
        
            
    
    
    datas = dl.DataLoader(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_iters = 0
    
    if datas.dataset.mode == "image":
        if opt.variational:
            model = VarVisAutoEncoder(opt.visual_aa,
                                   **opt.visual_aa_args)
            model = model.to(device)
        else:
            model = VisAutoEncoder(opt.visual_aa,
                                   **opt.visual_aa_args)
            model = model.to(device)    
    else:
        raise(AttributeError(f"{datas.dataset.mode} mode is not supported"))
        
    # currently defaulting to adam, will add more optimizer options later
    loss_f = create_loss(opt)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, opt.epoch_num + 1):
        print(f"Epoch {epoch}")
        epoch_start = time.time()
        
        iter_count = 0
        epoch_loss = 0    
        
        # sleep to ensure tqdm bars looks nice
        time.sleep(0.2)
        
        pbar = tqdm(total = len(datas))
        for i, data in enumerate(datas):
            
            data = data.to(device)

            # undoing accumulation of grad
            optimizer.zero_grad()

            batch_size = opt.batch_size
            total_iters += batch_size

            pbar.update(batch_size)
            
            # run data through model
            output = model.forward(data)
            
            # calculate loss
            loss = loss_f(output, data, total_iters, model, opt, device)
            
            if torch.isnan(loss):
                raise(RuntimeError("Loss is NaN. Consider increasing iterations before using KL Divergence loss"))
            
            # do backprop
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            # adjust weights with optimizer based on backprop
            optimizer.step()
            
            
            epoch_loss += loss.item()

        
        model.save()
        pbar.close()
        
        # sleep to ensure tqdm bars looks nice
        time.sleep(0.2)
        epoch_end = time.time()

        print(f"End of epoch {epoch} \n\tEpoch Loss : {epoch_loss} \n\tEpoch Time : {epoch_end - epoch_start}")
    return model
        