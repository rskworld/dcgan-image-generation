"""
DCGAN Trainer Module

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module handles the training loop for DCGAN, including:
- Adversarial training of generator and discriminator
- Loss calculation and optimization
- Model checkpointing
- Training progress visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from dcgan_model import Generator, Discriminator, weights_init


class DCGANTrainer:
    """
    Trainer class for DCGAN model
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, generator, discriminator, device, 
                 lr=0.0002, beta1=0.5, nz=100):
        """
        Initialize the trainer
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: torch device (cuda or cpu)
            lr: Learning rate for optimizers
            beta1: Beta1 hyperparameter for Adam optimizers
            nz: Size of input noise vector
        """
        self.device = device
        self.nz = nz
        
        # Initialize models
        self.netG = generator.to(device)
        self.netD = discriminator.to(device)
        
        # Apply weight initialization
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        # Initialize loss function
        self.criterion = nn.BCELoss()
        
        # Create batch of latent vectors for visualization
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        
        # Setup optimizers
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Training statistics
        self.G_losses = []
        self.D_losses = []
        self.iters = 0
    
    def train_epoch(self, dataloader, epoch, num_epochs):
        """
        Train for one epoch
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            num_epochs: Total number of epochs
        """
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            self.netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1.0, dtype=torch.float, device=self.device)
            # Forward pass real batch through D
            output = self.netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
            # Generate fake image batch with G
            fake = self.netG(noise)
            label.fill_(0.0)
            # Classify all fake batch with D
            output = self.netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizerD.step()
            
            ############################
            # (2) Update Generator network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            label.fill_(1.0)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())
            self.iters += 1
    
    def save_checkpoint(self, epoch, checkpoint_dir='checkpoints'):
        """
        Save model checkpoints
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            epoch: Current epoch number
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save generator
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.netG.state_dict(),
            'optimizer_state_dict': self.optimizerG.state_dict(),
            'losses': self.G_losses,
        }, os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth'))
        
        # Save discriminator
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.netD.state_dict(),
            'optimizer_state_dict': self.optimizerD.state_dict(),
            'losses': self.D_losses,
        }, os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))
    
    def load_checkpoint(self, generator_path, discriminator_path):
        """
        Load model checkpoints
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            generator_path: Path to generator checkpoint
            discriminator_path: Path to discriminator checkpoint
        """
        if os.path.exists(generator_path):
            checkpoint = torch.load(generator_path, map_location=self.device)
            self.netG.load_state_dict(checkpoint['model_state_dict'])
            self.optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'losses' in checkpoint:
                self.G_losses = checkpoint['losses']
            print(f'Loaded generator from {generator_path}')
        
        if os.path.exists(discriminator_path):
            checkpoint = torch.load(discriminator_path, map_location=self.device)
            self.netD.load_state_dict(checkpoint['model_state_dict'])
            self.optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'losses' in checkpoint:
                self.D_losses = checkpoint['losses']
            print(f'Loaded discriminator from {discriminator_path}')
    
    def generate_samples(self, num_samples=64, noise=None):
        """
        Generate sample images
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            num_samples: Number of samples to generate
            noise: Optional noise tensor, if None, random noise will be generated
            
        Returns:
            Generated images tensor
        """
        self.netG.eval()
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
            fake = self.netG(noise)
        self.netG.train()
        return fake

