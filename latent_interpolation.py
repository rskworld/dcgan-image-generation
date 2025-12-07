"""
Latent Space Interpolation Utilities

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides utilities for interpolating in the latent space
to generate smooth transitions between images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def linear_interpolation(z1, z2, num_steps=10):
    """
    Linear interpolation between two latent vectors
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        z1: First latent vector
        z2: Second latent vector
        num_steps: Number of interpolation steps
    
    Returns:
        Interpolated latent vectors
    """
    alphas = np.linspace(0, 1, num_steps)
    interpolated = []
    
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        interpolated.append(z)
    
    return torch.stack(interpolated)


def spherical_interpolation(z1, z2, num_steps=10):
    """
    Spherical interpolation (SLERP) between two latent vectors
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        z1: First latent vector
        z2: Second latent vector
        num_steps: Number of interpolation steps
    
    Returns:
        Interpolated latent vectors
    """
    # Normalize vectors
    z1_norm = z1 / torch.norm(z1)
    z2_norm = z2 / torch.norm(z2)
    
    # Calculate angle
    dot = torch.clamp(torch.sum(z1_norm * z2_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    
    alphas = np.linspace(0, 1, num_steps)
    interpolated = []
    
    for alpha in alphas:
        if theta < 1e-6:
            z = (1 - alpha) * z1 + alpha * z2
        else:
            z = (torch.sin((1 - alpha) * theta) * z1_norm + 
                 torch.sin(alpha * theta) * z2_norm) / torch.sin(theta)
            z = z * torch.norm(z1)  # Scale back
        interpolated.append(z)
    
    return torch.stack(interpolated)


def generate_interpolation(generator, z1, z2, num_steps=10, 
                          interpolation_type='linear', device='cuda'):
    """
    Generate images from interpolated latent vectors
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        generator: Trained generator model
        z1: First latent vector
        z2: Second latent vector
        num_steps: Number of interpolation steps
        interpolation_type: 'linear' or 'spherical'
        device: torch device
    
    Returns:
        Generated images
    """
    generator.eval()
    
    if interpolation_type == 'linear':
        z_interp = linear_interpolation(z1, z2, num_steps)
    else:
        z_interp = spherical_interpolation(z1, z2, num_steps)
    
    z_interp = z_interp.to(device)
    
    with torch.no_grad():
        images = generator(z_interp)
    
    generator.train()
    
    return images.cpu()


def visualize_interpolation(generator, z1, z2, num_steps=10,
                           interpolation_type='linear', device='cuda',
                           save_path=None):
    """
    Visualize interpolation between two latent vectors
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        generator: Trained generator model
        z1: First latent vector
        z2: Second latent vector
        num_steps: Number of interpolation steps
        interpolation_type: 'linear' or 'spherical'
        device: torch device
        save_path: Path to save visualization
    """
    images = generate_interpolation(
        generator, z1, z2, num_steps, interpolation_type, device
    )
    
    # Create grid
    grid = make_grid(images, nrow=num_steps, normalize=True, value_range=(-1, 1))
    grid_np = grid.numpy().transpose((1, 2, 0))
    grid_np = np.clip(grid_np, 0, 1)
    
    plt.figure(figsize=(15, 2))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(f'{interpolation_type.capitalize()} Interpolation')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Interpolation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_latent_walk(generator, nz=100, num_images=10, device='cuda',
                        save_path=None):
    """
    Generate a walk through latent space
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        generator: Trained generator model
        nz: Size of latent vector
        num_images: Number of images to generate
        device: torch device
        save_path: Path to save visualization
    """
    generator.eval()
    
    # Start with random noise
    z_start = torch.randn(1, nz, 1, 1, device=device)
    z_end = torch.randn(1, nz, 1, 1, device=device)
    
    # Generate interpolation
    images = generate_interpolation(generator, z_start, z_end, num_images, 'linear', device)
    
    # Create grid
    grid = make_grid(images, nrow=num_images, normalize=True, value_range=(-1, 1))
    grid_np = grid.numpy().transpose((1, 2, 0))
    grid_np = np.clip(grid_np, 0, 1)
    
    plt.figure(figsize=(15, 2))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title('Latent Space Walk')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Latent walk saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    generator.train()

