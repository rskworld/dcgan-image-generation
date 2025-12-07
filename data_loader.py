"""
Data Loading Utilities for DCGAN

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides utilities for loading and preprocessing image datasets
for DCGAN training, including support for custom datasets and common datasets
like CelebA, CIFAR-10, etc.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os


class ImageDataset(Dataset):
    """
    Custom dataset class for loading images from a directory
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset
        
        Args:
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # Get all image files
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.image_files.append(os.path.join(root_dir, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return dummy label


def get_dataloader(dataset_name='custom', root='./data', image_size=64, 
                   batch_size=128, num_workers=2, custom_dir=None):
    """
    Get DataLoader for specified dataset
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        dataset_name: Name of dataset ('custom', 'celeba', 'cifar10', 'mnist')
        root: Root directory for datasets
        image_size: Size to resize images to
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading
        custom_dir: Directory for custom dataset (required if dataset_name='custom')
    
    Returns:
        DataLoader object
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Load dataset
    if dataset_name == 'custom':
        if custom_dir is None:
            raise ValueError("custom_dir must be provided for custom dataset")
        dataset = ImageDataset(custom_dir, transform=transform)
    elif dataset_name == 'celeba':
        dataset = datasets.CelebA(root=root, split='train', download=True, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    elif dataset_name == 'mnist':
        # MNIST is grayscale, so we need different transform
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Single channel
        ])
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return dataloader


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1] for visualization
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        tensor: Normalized tensor
    
    Returns:
        Denormalized tensor
    """
    return (tensor + 1) / 2

