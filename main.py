"""
Main training script for DCGAN

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script can be used to train DCGAN from the command line.
Run: python main.py
"""

import torch
import os
from dcgan_model import Generator, Discriminator
from trainer import DCGANTrainer
from data_loader import get_dataloader
from utils import generate_and_save_images, plot_training_losses
import config


def main():
    """
    Main training function
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("=" * 50)
    print("DCGAN Training Script")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if config.SAVE_CHECKPOINTS:
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {config.DATASET_NAME}")
    dataloader = get_dataloader(
        dataset_name=config.DATASET_NAME,
        root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        custom_dir=config.CUSTOM_DATA_DIR if config.DATASET_NAME == 'custom' else None
    )
    print(f"Dataset loaded. Number of batches: {len(dataloader)}")
    
    # Create models
    print("\nCreating Generator and Discriminator...")
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
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    # Create trainer
    trainer = DCGANTrainer(
        generator=netG,
        discriminator=netD,
        device=device,
        lr=config.LR,
        beta1=config.BETA1,
        nz=config.NZ
    )
    
    # Training loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("-" * 50)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        trainer.train_epoch(dataloader, epoch, config.NUM_EPOCHS)
        
        # Save checkpoint
        if config.SAVE_CHECKPOINTS and epoch % config.CHECKPOINT_INTERVAL == 0:
            trainer.save_checkpoint(epoch, config.CHECKPOINT_DIR)
            print(f"Checkpoint saved at epoch {epoch}")
        
        # Generate and save sample images
        if epoch % 1 == 0:  # Save images every epoch
            sample_path = os.path.join(config.OUTPUT_DIR, f'epoch_{epoch:03d}_samples.png')
            generate_and_save_images(
                trainer.netG,
                device,
                nz=config.NZ,
                num_images=config.NUM_VISUALIZATION_SAMPLES,
                filepath=sample_path,
                noise=trainer.fixed_noise
            )
            print(f"Sample images saved to {sample_path}")
    
    # Save final checkpoint
    if config.SAVE_CHECKPOINTS:
        trainer.save_checkpoint(config.NUM_EPOCHS, config.CHECKPOINT_DIR)
    
    # Plot training losses
    if config.PLOT_LOSSES:
        loss_plot_path = os.path.join(config.OUTPUT_DIR, 'training_losses.png')
        plot_training_losses(trainer.G_losses, trainer.D_losses, loss_plot_path)
        print(f"\nTraining losses plot saved to {loss_plot_path}")
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()

