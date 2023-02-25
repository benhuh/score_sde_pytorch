
import torch
# import functools
# from torch.optim import Adam, AdamW
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from torchvision.datasets import MNIST
# from torchvision.utils import make_grid
# import tqdm

import torch.nn as nn
# import torch.nn.functional as F
import numpy as np



def GroupNorm_or_Identity(*args, identity=False, **kwargs):
    return torch.nn.Identity() if identity else GroupNorm(*args, **kwargs)
        
class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, *args, bias=False, weight=True, **kwargs): #num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
        super().__init__(*args, **kwargs)
        if bias==False:
            self.bias = None 
        if weight==False:
            self.weight = None 

            
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim, bias):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        
    def forward(self, x):
        return self.dense(x)[..., None, None]

#@title Defining a time-dependent score-based model (double click to expand or collapse)

class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, channels, embed_dim, bias, group_norm_args, **kwargs):  
    """Initialize a time-dependent score-based network.

    Args:
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    group_norm_args = group_norm_args or dict()
    
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),  nn.Linear(embed_dim, embed_dim))
    
    self.dense1 = Dense(embed_dim, channels[0], bias=bias)
    self.dense2 = Dense(embed_dim, channels[1], bias=bias)
    self.dense3 = Dense(embed_dim, channels[2], bias=bias)
    self.dense4 = Dense(embed_dim, channels[3], bias=bias)

    self.dense5 = Dense(embed_dim, channels[2], bias=bias)
    self.dense6 = Dense(embed_dim, channels[1], bias=bias)
    self.dense7 = Dense(embed_dim, channels[0], bias=bias)

    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.gnorm1 = GroupNorm_or_Identity(4, num_channels=channels[0], **group_norm_args)
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.gnorm2 = GroupNorm_or_Identity(32, num_channels=channels[1], **group_norm_args)
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.gnorm3 = GroupNorm_or_Identity(32, num_channels=channels[2], **group_norm_args)
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.gnorm4 = GroupNorm_or_Identity(32, num_channels=channels[3], **group_norm_args)    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.tgnorm4 = GroupNorm_or_Identity(32, num_channels=channels[2], **group_norm_args)
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.tgnorm3 = GroupNorm_or_Identity(32, num_channels=channels[1], **group_norm_args)
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.tgnorm2 = GroupNorm_or_Identity(32, num_channels=channels[0], **group_norm_args)
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    # self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    # Encoding path
    h1 = self.conv1(x)    
    h1 += self.dense1(embed)
    ## Incorporate information from t
    ## Group normalization
    h1 = self.act(self.gnorm1(h1))
    h2 = self.conv2(h1) + self.dense2(embed)
    h2 = self.act(self.gnorm2(h2))
    h3 = self.conv3(h2) + self.dense3(embed)
    h3 = self.act(self.gnorm3(h3))
    h4 = self.conv4(h3) + self.dense4(embed)
    h4 = self.act(self.gnorm4(h4))

    # Decoding path
    h = self.tconv4(h4) + self.dense5(embed)    ## Skip connection from the encoding path
    h = self.act(self.tgnorm4(h))
    h = self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)
    h = self.act(self.tgnorm3(h))
    h = self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)
    h = self.act(self.tgnorm2(h))
    h = self.tconv1(torch.cat([h, h1], dim=1))
    return h  # score * std   \approx - noise_est/std =  -(x_noisy - x) /std  = - z    # score = - grad_log_P 