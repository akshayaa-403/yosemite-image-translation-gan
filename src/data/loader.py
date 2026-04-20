"""Data loading utilities for CycleGAN"""
import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_data_loader(image_type, image_dir=r'/content/drive/MyDrive/summer2winter_yosemite', 
                    image_size=128, batch_size=16, num_workers=0):
    """
    Load training and test data for a given image type.
    
    Args:
        image_type (str): Type of images ('summer' or 'winter')
        image_dir (str): Base directory containing image folders
        image_size (int): Size to resize images to
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes
    
    Returns:
        tuple: (train_loader, test_loader) DataLoaders
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    image_path = image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, test_loader
