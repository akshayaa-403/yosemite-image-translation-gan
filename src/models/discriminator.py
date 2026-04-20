"""Discriminator model for CycleGAN"""
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Create a convolutional block with optional batch normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolutional kernel
        stride (int): Stride for convolution
        padding (int): Padding for convolution
        batch_norm (bool): Whether to apply batch normalization
    
    Returns:
        nn.Sequential: Sequential model with conv and optional batch norm
    """
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels,
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=False
    )
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """
    Discriminator network for CycleGAN.
    
    Classifies whether an image is real or fake.
    """
    
    def __init__(self, conv_dim=64):
        """
        Initialize Discriminator.
        
        Args:
            conv_dim (int): Base number of convolutional filters
        """
        super(Discriminator, self).__init__()

        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.conv5 = conv(conv_dim*8, 1, 8, stride=1, padding=0, batch_norm=False)

    def forward(self, x):
        """
        Forward pass through discriminator.
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Discriminator output (real/fake classification)
        """
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out)
        
        return out
