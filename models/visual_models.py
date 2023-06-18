import torch.nn as nn
from torch import randn, save, exp
from torch.distributions import Normal
from torchvision import datasets, transforms


class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers and a skip connection."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size = 7, 
                 dilation=1, 
                 bias=True, 
                 activation=nn.LeakyReLU(negative_slope=0.3),
                 dropout = 0.0,):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, padding = "same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, bias=bias)

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        x_p = self.activation(self.conv1(x))
        x_p = self.conv2(x_p)
        x_p = self.dropout(x_p)
        return x + x_p
    
class DownsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 scale_factor = 4,
                 dropout = 0.0):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv = nn.Conv2d(in_channels, out_channels, 
                              stride = scale_factor, 
                              kernel_size = kernel_size, 
                              padding = kernel_size // 2)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x
    
class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 scale_factor = 4,
                 dropout = 0.0):
        super().__init__()
        self.scale_factor = scale_factor

        # this style of upsample avoids checkerboard artifacts
        self.upsample = nn.Upsample(scale_factor = scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size = kernel_size,
                              padding = "same")
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self,
                 n_blocks = 4,
                 in_channels=3,
                 in_size=(256, 256),
                 dropout_val = 0.2,
                 inner_linear = True,
                 latent_dim = 512,
                 channel_scale = 2,
                 norm = nn.LayerNorm,
                 activation = nn.GELU(),
                 downsample_factor = 4):
        
        super().__init__()

        self.n_blocks = n_blocks
        self.in_size = in_size
        ds_ratio = downsample_factor ** n_blocks
        self.out_size = (in_size[0] // ds_ratio, in_size[1] // ds_ratio)
        self.in_channels = in_channels
        self.dropout_val = dropout_val
        self.inner_linear = inner_linear
        self.norm = norm

        encoder = []
        channels = in_channels
        for i in range(n_blocks):
            new_channels = channels * channel_scale
            encoder.append(DownsampleBlock(channels, 
                                           new_channels, 
                                           dropout = dropout_val,
                                           scale_factor = downsample_factor))
            encoder.append(ResidualBlock(new_channels, 
                                         new_channels, 
                                         dropout = dropout_val))
            encoder.append(activation)
            channels = new_channels

        self.final_channels = channels
        
        if inner_linear:
            #TODO : make sure decoder can get this shape to unflatten
            encoder.append(nn.Flatten())
            encoder.append(nn.Linear(channels * self.out_size[0] * self.out_size[1], latent_dim))
            encoder.append(activation)

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,
                 n_blocks = 4,
                 in_channels=24,
                 in_size=(4, 4),
                 dropout_val = 0.2,
                 inner_linear = True,
                 latent_dim = 512,
                 channel_scale = 2,
                 norm = nn.LayerNorm,
                 activation = nn.GELU(),
                 upsample_factor = 4,
                 final_activation = nn.Tanh()):
        
        super().__init__()

        self.n_blocks = n_blocks
        self.in_size = in_size
        us_ratio = upsample_factor ** n_blocks
        self.out_size = (in_size[0] * us_ratio, in_size[1] * us_ratio)
        self.in_channels = in_channels
        self.dropout_val = dropout_val
        self.inner_linear = inner_linear
        self.norm = norm

        decoder = []

        if inner_linear:
            decoder.append(nn.Linear(latent_dim, in_channels * self.in_size[0] * self.in_size[1]))
            decoder.append(nn.Unflatten(-1, (in_channels, self.in_size[0], self.in_size[1])))
            decoder.append(activation)

        channels = in_channels
        for i in range(n_blocks):
            new_channels = channels // channel_scale
            decoder.append(activation)
            decoder.append(UpsampleBlock(channels, 
                                         new_channels, 
                                         dropout = dropout_val,
                                         scale_factor = upsample_factor))
            decoder.append(ResidualBlock(new_channels, new_channels, dropout = dropout_val))
            
            channels = new_channels

        decoder.append(final_activation)

        self.final_channels = channels
        
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.decoder(x)
        return x


class VisAutoEncoder(nn.Module):
    """
    The standard convolutional autoencoder used in this project. 
    Parameters:
    ----------

    """
    def __init__(self,
                 n_blocks = 4,
                 name = "default_model",
                 in_channels = 3,
                 in_size = (256, 256),
                 dropout_val = 0.2,
                 inner_linear = True,
                 latent_dim = 512,
                 norm = nn.LayerNorm):
        super(VisAutoEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.in_size = in_size
        self.in_channels = in_channels
        self.dropout_val = dropout_val
        self.name = name
        self.inner_linear = inner_linear
        self.latent_dim = latent_dim
        self.norm = norm

        self.encoder = Encoder(n_blocks = n_blocks,
                               in_channels = in_channels,
                               in_size = in_size,
                               dropout_val = dropout_val,
                               inner_linear = inner_linear,
                               norm = norm,
                               latent_dim = latent_dim)
        
        self.decoder = Decoder(n_blocks = n_blocks,
                               in_channels = self.encoder.final_channels,
                               in_size = self.encoder.out_size,
                               dropout_val = dropout_val,
                               inner_linear = inner_linear,
                               norm = norm,
                               latent_dim = latent_dim)
        
        
        print(f"Generating Visual Autoencoder with parameters:\n\tName : {name}\n\tImage (C,H,W) : ({in_channels}, {in_size})\n\tWith dropout value {dropout_val}")

        self.print_self_params()
        print(f"Autoencoder Latent Dimension Size: {latent_dim}")

    def forward(self, x, return_everything = True):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def print_self_params(self):
        print("Total Model Parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    # wrapper for image encoding
    def encode_image(self, image):
        encoding = self.encoder(image)
        encoding_vector = encoding.detach()
        return encoding_vector
    
    # simple wrapper for saving
    def save(self, save_path = "ckpts/"):
        save(self.state_dict(), save_path + self.name + ".pt")
        print(f"\tModel saved to {save_path}")
    
            
            
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
                 n_blocks = 3,
                 name = "default_model",
                 in_channels=3,
                 in_size=(256, 256),
                 dropout_val = 0.2,
                 latent_dim = 512,
                 norm = nn.LayerNorm):
        super().__init__()

        self.add_module("mean_layer", nn.Linear(latent_dim, latent_dim))
        self.add_module("var_layer", nn.Linear(latent_dim, latent_dim))

    def forward(self, x, return_everything = False):
        #TODO : check this
        y = self.encoder(x)
        mean = self.mean_layer(y)
        log_variance = self.var_layer(y)
        variance = exp(log_variance / 2)
        pmf = Normal(mean, variance)
        h = pmf.rsample()
        x_out = self.decoder(h)
        if return_everything:
            return x, mean, log_variance, h, x_out
        else:
            return x_out
    
    def encode_image(self, image):
        encoding = self.encoder(image)

        mean = self.mean_layer(encoding)
        log_variance = self.var_layer(encoding)
        variance = exp(log_variance)

        pmf = Normal(mean, variance)
        encoding = pmf.rsample()
        
        encoding_vector = encoding.detach()
        return encoding_vector


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tests = [VisAutoEncoder(), VarVisAutoEncoder()]
    for test in tests:
        test_im = randn(1, 3, 256, 256)
        test_result = test(test_im).detach()
        print("Shape stayed same:", test_result.shape == test_im.shape)
        plt.imshow(test_im[0].permute(1, 2, 0))
        plt.imshow(test_result[0].permute(1, 2, 0))

    