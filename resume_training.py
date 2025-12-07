"""
Resume Training Utilities

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides utilities for resuming training from checkpoints.
"""

import torch
import os
from dcgan_model import Generator, Discriminator
from trainer import DCGANTrainer
import config


def load_training_state(checkpoint_dir, generator, discriminator, 
                      optimizer_g, optimizer_d, device):
    """
    Load complete training state from checkpoint
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        device: torch device
    
    Returns:
        Dictionary with training state (epoch, losses, etc.)
    """
    # Find latest checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('generator_epoch_') and f.endswith('.pth')]
    
    if not checkpoint_files:
        print("No checkpoints found")
        return None
    
    # Extract epoch numbers and find latest
    epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoint_files]
    latest_epoch = max(epochs)
    
    gen_path = os.path.join(checkpoint_dir, f'generator_epoch_{latest_epoch}.pth')
    disc_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{latest_epoch}.pth')
    
    # Load generator
    if os.path.exists(gen_path):
        checkpoint = torch.load(gen_path, map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded generator from epoch {latest_epoch}")
    
    # Load discriminator
    if os.path.exists(disc_path):
        checkpoint = torch.load(disc_path, map_location=device)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded discriminator from epoch {latest_epoch}")
    
    # Return training state
    state = {
        'epoch': latest_epoch,
        'g_losses': checkpoint.get('losses', []),
        'd_losses': checkpoint.get('losses', [])
    }
    
    return state


def resume_training(checkpoint_dir, trainer, start_epoch=None):
    """
    Resume training from checkpoint
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        trainer: DCGANTrainer instance
        start_epoch: Starting epoch (None to auto-detect)
    
    Returns:
        Starting epoch number
    """
    if start_epoch is None:
        # Find latest checkpoint
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                           if f.startswith('generator_epoch_') and f.endswith('.pth')]
        
        if checkpoint_files:
            epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoint_files]
            start_epoch = max(epochs) + 1
        else:
            start_epoch = 1
    else:
        start_epoch = start_epoch + 1
    
    # Load checkpoints
    gen_path = os.path.join(checkpoint_dir, f'generator_epoch_{start_epoch-1}.pth')
    disc_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{start_epoch-1}.pth')
    
    if os.path.exists(gen_path) and os.path.exists(disc_path):
        trainer.load_checkpoint(gen_path, disc_path)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"No checkpoint found for epoch {start_epoch-1}, starting from beginning")
        start_epoch = 1
    
    return start_epoch


def create_resume_script(checkpoint_dir, num_epochs, config_file='config.py'):
    """
    Create a script to resume training
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        num_epochs: Total number of epochs
        config_file: Path to config file
    """
    script_content = f'''"""
Resume Training Script

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script resumes training from the latest checkpoint.
"""

import torch
import os
from dcgan_model import Generator, Discriminator
from trainer import DCGANTrainer
from data_loader import get_dataloader
from utils import generate_and_save_images, plot_training_losses
from resume_training import resume_training
import {config_file.replace('.py', '')} as config


def main():
    """
    Main function to resume training
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("=" * 50)
    print("DCGAN Resume Training")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    device = torch.device(config.DEVICE)
    print(f"Using device: {{device}}")
    
    # Create directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load dataset
    dataloader = get_dataloader(
        dataset_name=config.DATASET_NAME,
        root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        custom_dir=config.CUSTOM_DATA_DIR if config.DATASET_NAME == 'custom' else None
    )
    
    # Create models
    netG = Generator(
        nz=config.NZ,
        ngf=config.NGF,
        nc=config.NC,
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    netD = Discriminator(
        nc=config.NC,
        ndf=config.NDF,
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    # Create trainer
    trainer = DCGANTrainer(
        generator=netG,
        discriminator=netD,
        device=device,
        lr=config.LR,
        beta1=config.BETA1,
        nz=config.NZ
    )
    
    # Resume training
    start_epoch = resume_training(config.CHECKPOINT_DIR, trainer)
    
    # Continue training
    for epoch in range(start_epoch, num_epochs + 1):
        trainer.train_epoch(dataloader, epoch, num_epochs)
        
        if config.SAVE_CHECKPOINTS and epoch % config.CHECKPOINT_INTERVAL == 0:
            trainer.save_checkpoint(epoch, config.CHECKPOINT_DIR)
        
        if epoch % 1 == 0:
            sample_path = os.path.join(config.OUTPUT_DIR, f'epoch_{{epoch:03d}}_samples.png')
            generate_and_save_images(
                trainer.netG,
                device,
                nz=config.NZ,
                num_images=config.NUM_VISUALIZATION_SAMPLES,
                filepath=sample_path,
                noise=trainer.fixed_noise
            )
    
    print("\\nTraining completed!")


if __name__ == '__main__':
    main()
'''
    
    script_path = 'resume_training_script.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Resume script created: {script_path}")
    return script_path

