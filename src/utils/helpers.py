"""Helper utility functions"""
import torch
import torchvision
import matplotlib.pyplot as plt


def scale(x, feature_range=(-1, 1)):
    """
    Scale image tensor to a specified feature range.
    
    Args:
        x (torch.Tensor): Input tensor with values in [0, 1]
        feature_range (tuple): Target range (min, max)
    
    Returns:
        torch.Tensor: Scaled tensor
    """
    min_val, max_val = feature_range
    x = x * (max_val - min_val) + min_val
    return x


def save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, sample_dir='samples'):
    """
    Save sample translations during training.
    
    Args:
        epoch (int): Current training epoch
        fixed_Y (torch.Tensor): Fixed batch of Y domain images
        fixed_X (torch.Tensor): Fixed batch of X domain images
        G_YtoX (nn.Module): Generator from Y to X domain
        G_XtoY (nn.Module): Generator from X to Y domain
        batch_size (int): Batch size
        sample_dir (str): Directory to save samples
    """
    import os
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        fake_X = G_YtoX(fixed_Y)
        fake_Y = G_XtoY(fixed_X)
    
    # Create grid of samples
    X_samples = torch.cat([fixed_X, fake_X], dim=0)
    Y_samples = torch.cat([fixed_Y, fake_Y], dim=0)
    
    # Unscale from [-1, 1] to [0, 1]
    X_samples = (X_samples + 1) / 2
    Y_samples = (Y_samples + 1) / 2
    
    # Create and save combined image
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    grid_X = torchvision.utils.make_grid(X_samples, nrow=batch_size)
    grid_Y = torchvision.utils.make_grid(Y_samples, nrow=batch_size)
    
    grid_X_np = grid_X.permute(1, 2, 0).cpu().numpy()
    grid_Y_np = grid_Y.permute(1, 2, 0).cpu().numpy()
    
    axes[0].imshow(grid_X_np)
    axes[0].set_title(f'X to Y Translation (Epoch {epoch})')
    axes[0].axis('off')
    
    axes[1].imshow(grid_Y_np)
    axes[1].set_title(f'Y to X Translation (Epoch {epoch})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    sample_path = os.path.join(sample_dir, f'sample-{epoch:06d}.png')
    plt.savefig(sample_path)
    plt.close()


def checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y, checkpoint_dir='checkpoints_cyclegan'):
    """
    Save model checkpoint.
    
    Args:
        epoch (int): Current training epoch
        G_XtoY (nn.Module): Generator from X to Y
        G_YtoX (nn.Module): Generator from Y to X
        D_X (nn.Module): Discriminator for X domain
        D_Y (nn.Module): Discriminator for Y domain
        checkpoint_dir (str): Directory to save checkpoints
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'G_XtoY_state_dict': G_XtoY.state_dict(),
        'G_YtoX_state_dict': G_YtoX.state_dict(),
        'D_X_state_dict': D_X.state_dict(),
        'D_Y_state_dict': D_Y.state_dict(),
    }, checkpoint_path)
