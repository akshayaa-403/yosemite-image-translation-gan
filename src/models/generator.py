"""Generator models for CycleGAN"""
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


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Create a transposed convolutional block with optional batch normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolutional kernel
        stride (int): Stride for convolution
        padding (int): Padding for convolution
        batch_norm (bool): Whether to apply batch normalization
    
    Returns:
        nn.Sequential: Sequential model with transposed conv and optional batch norm
    """
    layers = []
    layers.append(nn.ConvTranspose2d(
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        bias=False
    ))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """
    Residual block for generator network.
    
    Implements skip connection for improved gradient flow.
    """
    
    def __init__(self, conv_dim):
        """
        Initialize ResidualBlock.
        
        Args:
            conv_dim (int): Number of convolutional filters
        """
        super(ResidualBlock, self).__init__()

        self.conv_layer1 = conv(
            in_channels=conv_dim, 
            out_channels=conv_dim,
            kernel_size=3, 
            stride=1, 
            padding=1, 
            batch_norm=True
        )

        self.conv_layer2 = conv(
            in_channels=conv_dim, 
            out_channels=conv_dim,
            kernel_size=3, 
            stride=1, 
            padding=1, 
            batch_norm=True
        )

    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        
        return out_2


class CycleGenerator(nn.Module):
    """
    Generator network for CycleGAN.
    
    Translates images from one domain to another using residual blocks.
    """
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        """
        Initialize CycleGenerator.
        
        Args:
            conv_dim (int): Base number of convolutional filters
            n_res_blocks (int): Number of residual blocks
        """
        super(CycleGenerator, self).__init__()

        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*4))

        self.res_blocks = nn.Sequential(*res_layers)

        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        """
        Forward pass through generator.
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Translated image tensor
        """
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.res_blocks(out)

        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.tanh(self.deconv3(out))

        return out
