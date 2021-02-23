import torch.nn as nn
from torch import randn, save, exp
from torch.distributions import Normal
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def create_final_conv(image_sizes):
    """
    This function generates a convolution with h,w = 1. Used the formula from
    here :https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    ----------
    image_sizes : the h,w of the output of each convolution layer in a list
    """
    h_prev = image_sizes[-1][0]
    w_prev = image_sizes[-1][1]
    padding = 0
    stride = 1
    kernel = h_prev
    # returning 1 for h_out, w_out
    return 1, 1, padding, kernel, stride

class VisAutoEncoder(nn.Module):
    """
    The standard convolutional autoencoder used in this project. 
    Parameters:
    ----------
    filter_channel_sizes : list of tuples of form 
        (in_channels, out_channels, kernel_size, stride, padding)
        for each layer in the encoder. See test_autoencoder() for example
        Number of output channels for each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 256)
        Spatial size of the expected input image.
    dropout : bool, default True
        Enables or disables dropout in the encoder.
    dropout_val : float, default 0.2
        Proportion of tensor that will be dropped.
    """
    def __init__(self,
                 filter_channel_sizes,
                 name = "default_model",
                 in_channels=3,
                 in_size=(256, 256),
                 dropout = True,
                 dropout_val = 0.2,
                 latent_dim = 256,
                 inner_linear = False):
        super(VisAutoEncoder, self).__init__()
        self.in_size = in_size
        self.in_channels = in_channels
        self.dropout = dropout
        self.dropout_val = dropout_val
        self.name = name
        self.latent_dim = latent_dim
        self.inner_linear = False

        self.encoder= nn.Sequential()
        self.encoder.add_module("instancenorm1", nn.InstanceNorm2d(num_features=in_channels))
        
        print(f"Generating Visual Autoencoder with parameters:\n\tName : {name}\n\tImage (C,H,W) : ({in_channels}, {in_size})\n\tDropout Enabled : {dropout}", dropout * f"with value {dropout_val}")
        
        # this will be used for deciding output_padding size when deconving
        # neccesary because Conv2d can map different sized inputs to the same
        # output size eg both 511 and 512 get mapped to 257 when stride 2 is used
        size_list = [[in_size[0], in_size[1]]]
        
        #these layers do convolution
        for i, filter_size in enumerate(filter_channel_sizes):
            in_channels=filter_size[0]
            out_channels=filter_size[1]
            kernel=filter_size[2]
            stride=filter_size[3]
            padding=filter_size[4]
            self.encoder.add_module("conv{}".format(i + 1), nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False))
            # computing the  output sizes for this convolution layer
            # for the first i use in_size, else use previous size
            if not i:
                # formulas from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
                h_out = ((in_size[0] + 2*padding - (kernel - 1) - 1) // stride) + 1
                w_out = ((in_size[1] + 2*padding - (kernel - 1) - 1) // stride) + 1
                size_list.append([h_out, w_out])
            else:
                h_out = ((h_out + 2*padding - (kernel - 1) - 1) // stride) + 1
                w_out = ((w_out + 2*padding - (kernel - 1) - 1) // stride) + 1
                size_list.append([h_out, w_out])
                
            #dropout for robustness, only in the convolution steps
            if dropout:
                self.encoder.add_module("dropout{}".format(i + 1), nn.Dropout(p=dropout_val))
            self.encoder.add_module("relu{}".format(i + 1), nn.ReLU(inplace=True))
            
        h_out, w_out, padding, kernel, stride = create_final_conv(size_list)
        #in channels is previous out channels
        in_channels = out_channels
        out_channels = latent_dim
        
        size_list.append([1, 1])
        self.encoder.add_module("conv_final", nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False))
        
        filter_channel_sizes.append([in_channels, out_channels, kernel, stride, padding])
        
        self.encoder.add_module("flatten", nn.Flatten())
        
        #reversing size_list to get the successive sizes for deconv h,w
        size_list.reverse()
        self.decoder = nn.Sequential()
        
        if self.inner_linear:
            conv_out_size = out_channels * h_out * w_out
            self.encoder.add_module("linear_enc", nn.Linear(conv_out_size, latent_dim))
            self.decoder.add_module("linear_dec", nn.Linear(latent_dim, conv_out_size))
        
        self.decoder.add_module("unflatten", nn.Unflatten(1, (out_channels, h_out, w_out)))
        #these layers reverse the previous convolution ops
        for i, filter_size in enumerate(reversed(filter_channel_sizes)):
            #flip channels when going in reverse
            in_channels=filter_size[1]
            out_channels=filter_size[0]
            stride = filter_size[3]
            kernel=filter_size[2]
            padding=filter_size[4]
            
            # computing the estimated output sizes for this deconv layer
            # formulas from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
            h_in = (size_list[i][0] - 1) * stride - 2 * padding + (kernel - 1) + 1
            w_in = (size_list[i][1] - 1) * stride - 2 * padding + (kernel - 1) + 1
            
            # to decide output_padding, compute size diff for no output_padding
            # and the original shape
            output_padding = size_list[i + 1][0] - h_in 

            #print(h_in, size_list[i + 1][0], output_padding)
            self.decoder.add_module("deconv{}".format(i + 1), nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_padding = output_padding,
                bias=False))
            self.decoder.add_module("relu{}".format(i + 1), nn.ReLU(inplace=True))
        print("Total Model Parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
        self.describe_params()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    # simple wrapper for saving
    def save(self, save_path = "ckpts/"):
        save(self.state_dict(), save_path + self.name + ".pt")
        print(f"\tModel saved to {save_path}")
    
    def describe_params(self):
        for name, params in self.named_parameters():
            print(name, params.data.shape)
            
            
class VarVisAutoEncoder(VisAutoEncoder):
    """
    The variational version of the standard convolutional autoencoder used in this project. 
    Parameters:
    ----------
    filter_channel_sizes : list of tuples of form 
        (in_channels, out_channels, kernel_size, stride, padding)
        for each layer in the encoder. See test_autoencoder() for example
        Number of output channels for each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 256)
        Spatial size of the expected input image.
    dropout : bool, default True
        Enables or disables dropout in the encoder.
    dropout_val : float, default 0.2
        Proportion of tensor that will be dropped.
    """
    def __init__(self,
                 filter_channel_sizes,
                 name = "default_var_model",
                 in_channels=3,
                 in_size=(256, 256),
                 dropout = True,
                 dropout_val = 0.2,
                 latent_dim = 256):
        super(VisAutoEncoder, self).__init__()

        self.add_module("mean_layer", nn.Linear(latent_dim, latent_dim))
        self.add_module("var_layer", nn.Linear(latent_dim, latent_dim))

    def forward(self, x):
        y = self.encoder(x)
        mean = self.mean_layer(y)
        log_variance = self.var_layer(y)
        variance = exp(log_variance / 2)
        pmf = Normal(mean, variance)
        h = pmf.rsample()
        x_out = self.decoder(h)
        return x, mean, log_variance, h, x_out




    
def test_autoencoder():
    """
    Test function that shows how to construct an autoencoder from the class,
    and to display the before/after images with imshow
    """
    
    #first tuple in list says 3 input channels (RGB), 128 out channels, kernel size of 7, stride of 1, and padding 4
    #first item in a tuple will always have same size as previous tuple's output 
    test = VisAutoEncoder([(3, 128, 7, 1, 4), (128, 256, 3, 2, 2), (256, 512, 3, 2, 2)])
    test_im = randn(1, 3, 256, 256)
    test_result = test.forward(test_im).detach()
    print(test_result.shape)
    plt.imshow(test_im[0].permute(1, 2, 0))
    plt.imshow(test_result[0].permute(1, 2, 0))
    