"""
Example usage of DCGAN models

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This file demonstrates how to use the DCGAN models for training and inference.
"""

import torch
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dcgan_model import Generator, Discriminator, weights_init
from trainer import DCGANTrainer
from data_loader import get_dataloader
from utils import save_image_grid, plot_training_losses
import config


def example_training():
    """
    Example: Basic training setup
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("Example: Basic Training Setup")
    print("-" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    generator = Generator(
        nz=config.NZ,
        ngf=config.NGF,
        nc=config.NC,
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    discriminator = Discriminator(
        nc=config.NC,
        ndf=config.NDF,
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print("Models created successfully!")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")


def example_inference():
    """
    Example: Generate images from trained model
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("\nExample: Image Generation")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained generator
    checkpoint_path = 'checkpoints/final_generator.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return
    
    # Create generator
    generator = Generator(
        nz=config.NZ,
        ngf=config.NGF,
        nc=config.NC,
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    
    # Generate images
    num_samples = 64
    noise = torch.randn(num_samples, config.NZ, 1, 1, device=device)
    
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    
    # Save images
    output_path = 'outputs/example_generated.png'
    save_image_grid(fake_images, output_path)
    print(f"Generated images saved to {output_path}")


def example_custom_dataset():
    """
    Example: Using custom dataset
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("\nExample: Custom Dataset")
    print("-" * 50)
    
    # Load custom dataset
    custom_dir = './data/custom'
    
    if not os.path.exists(custom_dir):
        print(f"Custom dataset directory not found: {custom_dir}")
        print("Please create the directory and add your images.")
        return
    
    dataloader = get_dataloader(
        dataset_name='custom',
        root='./data',
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        custom_dir=custom_dir
    )
    
    print(f"Custom dataset loaded: {len(dataloader)} batches")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Total images: {len(dataloader) * config.BATCH_SIZE}")


def main():
    """
    Main function to run examples
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("=" * 50)
    print("DCGAN Usage Examples")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    # Run examples
    example_training()
    example_custom_dataset()
    example_inference()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()

