"""
Utility Functions for DCGAN

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module contains utility functions for visualization, image saving,
and other helper functions used throughout the DCGAN project.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os


def save_image_grid(images, filepath, nrow=8, normalize=True, value_range=(-1, 1)):
    """
    Save a grid of images to file
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        images: Tensor of images (B, C, H, W)
        filepath: Path to save the image
        nrow: Number of images per row in the grid
        normalize: Whether to normalize the images
        value_range: Range of values for normalization
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Create grid
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=value_range)
    
    # Convert to numpy and transpose
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Clip values to [0, 1]
    grid_np = np.clip(grid_np, 0, 1)
    
    # Save
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def plot_training_losses(G_losses, D_losses, filepath='training_losses.png'):
    """
    Plot training losses for generator and discriminator
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        G_losses: List of generator losses
        D_losses: List of discriminator losses
        filepath: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def generate_and_save_images(generator, device, nz=100, num_images=64, 
                             filepath='generated_samples.png', noise=None):
    """
    Generate images using the generator and save them
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        generator: Generator model
        device: torch device
        nz: Size of noise vector
        num_images: Number of images to generate
        filepath: Path to save the images
        noise: Optional noise tensor
    """
    generator.eval()
    with torch.no_grad():
        if noise is None:
            noise = torch.randn(num_images, nz, 1, 1, device=device)
        fake_images = generator(noise)
        save_image_grid(fake_images, filepath)
    generator.train()


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

