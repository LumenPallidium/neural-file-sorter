import time
import torch
import torchvision
import os
from tqdm import tqdm
from options.options import Options
import util.dataloader as dl
from models.visual_models import *
import models.audio_models
import numpy as np
import warnings

def l_n_reg(model, device, norm = 1):
    """
    Wrapper to add a weight regularizer, with adjustable norm.
    ----------
    model: model with the weights to be regularized
    device: either cpu or gpu
    norm: default 1, the norm to use in regularization
    """
    reg_loss = torch.tensor(0., requires_grad=True).to(device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, norm)
    return reg_loss

class WarmUpScheduler(object):
    def __init__(self, optimizer, scheduler, warmup_iter, total_iter = 300000):
        self.optimizer = optimizer
        self.scheduler = scheduler(optimizer, total_iter - warmup_iter, eta_min = optimizer.defaults["lr"] / 1e3)
        self.warmup_iter = warmup_iter
        self.iter = 0
    
    def step(self):
        if self.iter < self.warmup_iter:
            lr = self.iter / self.warmup_iter * self.scheduler.get_last_lr()[0]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
        self.iter += 1

def tensor2im(x):
    return torchvision.transforms.ToPILImage()(x)

def save_im(x, path):
    tensor2im(x).save(path)

def export_samples(samples, opt, suffix = "", rows = 2, cols = 2):
    sample_save_dir = os.path.join(opt.sample_dir, "samples")
    os.makedirs(sample_save_dir, exist_ok = True)

    if len(samples.shape) == 4:
        # select only the rows x cols item in the batch
        samples = samples[:rows*cols]
        # make a grid of the samples
        samples = torchvision.utils.make_grid(samples, nrow = rows, padding = 2, pad_value = 1)
    # note that rank 3 tensors are implictlty assumed to be (C, H, W)
    
    # save the samples
    save_im(samples, os.path.join(sample_save_dir, f"sample_{suffix}.png"))


def create_loss(opt):
    """
    Somewhat ugly but this works in a more efficient way than having a loss 
    function with many conditionals in it, since variational autoencoder
    returns multiple outputs but vanilla only retuns one.
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
        def loss_f(output, data, curr_iter, model, opt, device):
            # for a variational model, use KL Divergence for latent
            # probability distribution loss, and MSE for reconstruction loss
            loss_mse = torch.nn.MSELoss()
            
            # output of forward in this case is model input, mean/variance hidden layers
            # the encoded state as a sample, and the ouput state
            mean = output[1]
            log_variance = output[2]

            output = output[4]
            
            # VAE is often unstable initially, so vary the loss depending on iter
            if curr_iter > opt.vae_warmup_iters:
                # loss between the input and actual output
                loss = loss_mse(data, output)
            else:
                # warmup loss does not add variance to the output
                x_hat = model.decoder(mean)
                x_hat = model.out_conv(x_hat)
                loss = loss_mse(data, x_hat)

            # KL Div loss when goal distribution is assumed standard normal
            # and posterior approximation normal (not neccesarily standard)
            # https://arxiv.org/abs/1312.6114
            kl_loss = opt.beta * (-0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim = -1))
            loss += kl_loss.mean()

            if opt.regularize:
                reg_loss = l_n_reg(model, device, norm = opt.reg_norm)
                loss += opt.reg_penalty * reg_loss

            return loss

    else:
        def loss_f(output, data, curr_iter, model, opt, device):
            # currently defaulting to mse loss, will add more loss function options later
            loss_crit = torch.nn.MSELoss()
            loss = loss_crit(output, data)
            if opt.regularize:
                reg_loss = l_n_reg(model, device, norm = opt.reg_norm)
                loss = loss_f(output, data) + opt.reg_penalty * reg_loss
            return loss
    return loss_f

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
    os.makedirs("ckpts", exist_ok = True)
    datas = dl.DataLoader(opt, map_depth = opt.map_depth)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_iters = 0
    
    if datas.dataset.mode == "image":
        model = VarVisAutoEncoder(**opt.visual_aa_args) if opt.variational else VisAutoEncoder(**opt.visual_aa_args)
        if opt.continue_train and os.path.exists("ckpts/" + model.name + ".pt"):
            print(f"Loading existing model: {model.name}")
            model.load_state_dict(torch.load("ckpts/" + model.name + ".pt"))
        model = model.to(device)

    else:
        raise(AttributeError(f"{datas.dataset.mode} mode is not supported"))
        
    # currently defaulting to adam, will add more optimizer options later
    loss_f = create_loss(opt)

    scheduler = WarmUpScheduler(torch.optim.Adam(model.parameters(), 
                                                lr = opt.lr, 
                                                amsgrad = True), 
                            torch.optim.lr_scheduler.CosineAnnealingLR, 
                            warmup_iter = opt.warmup_iters // opt.batch_size,)

    for epoch in range(1, opt.epoch_num + 1):
        print(f"Epoch {epoch}")
        epoch_start = time.time()
        
        epoch_loss = 0    
        
        # sleep to ensure tqdm bars looks nice
        time.sleep(0.2)
        
        pbar = tqdm(total = len(datas))
        for i, data in enumerate(datas):
            
            data = data.to(device)

            scheduler.optimizer.zero_grad()

            batch_size = opt.batch_size
            total_iters += opt.batch_size
            pbar.update(batch_size)
            
            # run data through model
            output = model(data, return_everything = True)
            
            # calculate loss
            loss = loss_f(output, data, total_iters, model, opt, device)
            
            if torch.isnan(loss):
                raise(RuntimeError("Loss is NaN. Consider increasing iterations before using KL Divergence loss"))
            
            # do backprop
            loss.backward()
            
            if opt.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            
            # adjust weights with optimizer based on backprop
            scheduler.optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()

            if (i % opt.samples_every) == 0:
                model.save()
                if opt.variational:
                    out_im = output[4]
                else:
                    out_im = output
                # save samples
                export_samples(out_im, opt, suffix = str(i))
                print(f"\tSamples saved. Current loss per iter: {epoch_loss / (i + 1)}")

        model.save()
        pbar.close()
        
        # sleep to ensure tqdm bars looks nice
        time.sleep(0.2)
        epoch_end = time.time()

        print(f"End of epoch {epoch} \n\tEpoch Loss : {epoch_loss} \n\tEpoch Time : {epoch_end - epoch_start}")
    return model
        
if __name__ == "__main__":
    opt = Options()
    train(opt)