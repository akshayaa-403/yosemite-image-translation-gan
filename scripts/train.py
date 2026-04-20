"""Main training script for CycleGAN"""
import os
import torch
import torch.optim as optim
import argparse

from src.data import get_data_loader
from src.models import CycleGenerator, Discriminator
from src.losses import real_mse_loss, fake_mse_loss, cycle_consistency_loss
from src.utils import scale, save_samples, checkpoint


def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """
    Create and initialize all models.
    
    Args:
        g_conv_dim (int): Base conv dimension for generators
        d_conv_dim (int): Base conv dimension for discriminators
        n_res_blocks (int): Number of residual blocks in generators
    
    Returns:
        tuple: (G_XtoY, G_YtoX, D_X, D_Y)
    """
    G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)

    D_X = Discriminator(conv_dim=d_conv_dim)
    D_Y = Discriminator(conv_dim=d_conv_dim)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  G_XtoY, G_YtoX, D_X, D_Y,
                  g_optimizer, d_x_optimizer, d_y_optimizer,
                  n_epochs=1000, print_every=10, sample_every=100, checkpoint_every=1000):
    """
    Main training loop.
    
    Args:
        dataloader_X: Training data loader for X domain
        dataloader_Y: Training data loader for Y domain
        test_dataloader_X: Test data loader for X domain
        test_dataloader_Y: Test data loader for Y domain
        G_XtoY: Generator from X to Y
        G_YtoX: Generator from Y to X
        D_X: Discriminator for X domain
        D_Y: Discriminator for Y domain
        g_optimizer: Optimizer for generators
        d_x_optimizer: Optimizer for D_X
        d_y_optimizer: Optimizer for D_Y
        n_epochs: Number of training epochs
        print_every: Frequency of loss printing
        sample_every: Frequency of sample saving
        checkpoint_every: Frequency of checkpoint saving
    
    Returns:
        list: Training losses
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    losses = []
    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)
    fixed_X = next(test_iter_X)[0].to(device)
    fixed_Y = next(test_iter_Y)[0].to(device)
    fixed_X = scale(fixed_X).to(device)
    fixed_Y = scale(fixed_Y).to(device)

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(dataloader_X), len(dataloader_Y))

    for epoch in range(1, n_epochs + 1):
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = next(iter_X)
        images_X = scale(images_X).to(device)

        images_Y, _ = next(iter_Y)
        images_Y = scale(images_Y).to(device)

        # Train D_X
        d_x_optimizer.zero_grad()

        D_X_real_loss = real_mse_loss(D_X(images_X))
        G_Y2X_fake_image = G_YtoX(images_Y)
        D_X_fake_loss = fake_mse_loss(D_X(G_Y2X_fake_image))

        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()

        # Train D_Y
        D_Y_real_loss = real_mse_loss(D_Y(images_Y))
        G_X2Y_fake_image = G_XtoY(images_X)
        D_Y_fake_loss = fake_mse_loss(D_Y(G_X2Y_fake_image))

        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()

        # Train generators
        g_optimizer.zero_grad()

        G_X_img = G_YtoX(images_Y)
        G_X_real_loss = real_mse_loss(D_X(G_X_img))
        G_Y_reconstructed = G_XtoY(G_X_img)
        G_Y_consistency_loss = cycle_consistency_loss(images_Y, G_Y_reconstructed, 10)

        G_Y_img = G_XtoY(images_X)
        G_Y_real_loss = real_mse_loss(D_Y(G_Y_img))
        G_X_reconstructed = G_YtoX(G_Y_img)
        G_X_consistency_loss = cycle_consistency_loss(images_X, G_X_reconstructed, 10)

        g_total_loss = G_X_real_loss + G_Y_real_loss + G_Y_consistency_loss + G_X_consistency_loss
        g_total_loss.backward()
        g_optimizer.step()

        if epoch % print_every == 0:
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))

        if epoch % sample_every == 0:
            G_YtoX.eval()
            G_XtoY.eval()
            save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16)
            G_YtoX.train()
            G_XtoY.train()

        if epoch % checkpoint_every == 0:
            checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y)

    return losses


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train CycleGAN for image translation')
    parser.add_argument('--data_dir', type=str, default=r'/content/drive/MyDrive/summer2winter_yosemite',
                        help='Path to data directory')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=4000,
                        help='Number of training epochs')
    parser.add_argument('--g_conv_dim', type=int, default=64,
                        help='Base conv dimension for generators')
    parser.add_argument('--d_conv_dim', type=int, default=64,
                        help='Base conv dimension for discriminators')
    parser.add_argument('--n_res_blocks', type=int, default=6,
                        help='Number of residual blocks')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam optimizer')
    
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    dataloader_X, test_dataloader_X = get_data_loader(
        image_type='summer', 
        image_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
    dataloader_Y, test_dataloader_Y = get_data_loader(
        image_type='winter',
        image_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    # Create models
    print("Creating models...")
    G_XtoY, G_YtoX, D_X, D_Y = create_model(
        g_conv_dim=args.g_conv_dim,
        d_conv_dim=args.d_conv_dim,
        n_res_blocks=args.n_res_blocks
    )

    # Create optimizers
    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())
    g_optimizer = optim.Adam(g_params, args.lr, [args.beta1, args.beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(), args.lr, [args.beta1, args.beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(), args.lr, [args.beta1, args.beta2])

    # Train
    print("Starting training...")
    losses = training_loop(
        dataloader_X, dataloader_Y, 
        test_dataloader_X, test_dataloader_Y,
        G_XtoY, G_YtoX, D_X, D_Y,
        g_optimizer, d_x_optimizer, d_y_optimizer,
        n_epochs=args.num_epochs
    )

    print("Training complete!")


if __name__ == '__main__':
    main()
