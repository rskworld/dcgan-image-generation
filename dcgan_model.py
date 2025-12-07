"""
DCGAN (Deep Convolutional Generative Adversarial Network) Model Implementation

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module implements the DCGAN architecture with:
- Convolutional generator network
- Convolutional discriminator network
- Batch normalization and LeakyReLU activations
- Proper weight initialization for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    Custom weights initialization called on netG and netD
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator Network
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Architecture:
    - Input: Random noise vector (nz dimensions)
    - Output: Generated image (nc channels, image_size x image_size)
    - Uses transposed convolutions to upsample from noise to image
    - Batch normalization and ReLU activations
    """
    
    def __init__(self, nz=100, ngf=64, nc=3, image_size=64):
        """
        Initialize the Generator network
        
        Args:
            nz: Size of input noise vector
            ngf: Number of generator filters in the first conv layer
            nc: Number of channels in the training images
            image_size: Size of the generated images (assumed square)
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.image_size = image_size
        
        # Calculate number of upsampling layers needed
        # For 64x64: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        # For 128x128: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        if image_size == 64:
            layers = [
                # Input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # State size: (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # State size: (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # State size: (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # State size: (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # State size: (nc) x 64 x 64
            ]
        elif image_size == 128:
            layers = [
                # Input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                # State size: (ngf*16) x 4 x 4
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # State size: (ngf*8) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # State size: (ngf*4) x 16 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # State size: (ngf*2) x 32 x 32
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # State size: (ngf) x 64 x 64
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # State size: (nc) x 128 x 128
            ]
        else:
            raise ValueError(f"Unsupported image size: {image_size}. Supported sizes: 64, 128")
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, input):
        """
        Forward pass of the generator
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        """
        return self.main(input)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator Network
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Architecture:
    - Input: Image (nc channels, image_size x image_size)
    - Output: Probability that input is real (single value)
    - Uses strided convolutions to downsample from image to single value
    - Batch normalization and LeakyReLU activations
    """
    
    def __init__(self, nc=3, ndf=64, image_size=64):
        """
        Initialize the Discriminator network
        
        Args:
            nc: Number of channels in the training images
            ndf: Number of discriminator filters in the first conv layer
            image_size: Size of the input images (assumed square)
        """
        super(Discriminator, self).__init__()
        
        if image_size == 64:
            layers = [
                # Input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                # State size: 1 x 1 x 1
            ]
        elif image_size == 128:
            layers = [
                # Input is (nc) x 128 x 128
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf) x 64 x 64
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf*2) x 32 x 32
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf*4) x 16 x 16
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf*8) x 8 x 8
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # State size: (ndf*16) x 4 x 4
                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                # State size: 1 x 1 x 1
            ]
        else:
            raise ValueError(f"Unsupported image size: {image_size}. Supported sizes: 64, 128")
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, input):
        """
        Forward pass of the discriminator
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        """
        return self.main(input).view(-1, 1).squeeze(1)

