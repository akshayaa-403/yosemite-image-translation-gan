"""Loss functions for CycleGAN"""
import torch


def real_mse_loss(D_out):
    """
    MSE loss for real images (discriminator should output 1).
    
    Args:
        D_out (torch.Tensor): Discriminator output for real images
    
    Returns:
        torch.Tensor: MSE loss value
    """
    return torch.mean((D_out - 1) ** 2)


def fake_mse_loss(D_out):
    """
    MSE loss for fake images (discriminator should output 0).
    
    Args:
        D_out (torch.Tensor): Discriminator output for fake images
    
    Returns:
        torch.Tensor: MSE loss value
    """
    return torch.mean(D_out ** 2)


def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    """
    Cycle consistency loss (L1 loss).
    
    Ensures that when an image is translated to another domain and back,
    it matches the original image.
    
    Args:
        real_im (torch.Tensor): Original image
        reconstructed_im (torch.Tensor): Reconstructed image after cycle
        lambda_weight (float): Weight for the loss
    
    Returns:
        torch.Tensor: Weighted cycle consistency loss
    """
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    return lambda_weight * reconstr_loss
