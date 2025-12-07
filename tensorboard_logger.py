"""
TensorBoard Logger for DCGAN Training

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides TensorBoard logging for training visualization.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.utils import make_grid
import numpy as np


class TensorBoardLogger:
    """
    TensorBoard logger for DCGAN training
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, log_dir='runs/dcgan', comment=''):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory to save logs
            comment: Comment to add to log directory name
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        self.global_step = 0
    
    def log_scalar(self, tag, value, step=None):
        """
        Log scalar value
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            tag: Tag name
            value: Scalar value
            step: Step number (uses global_step if None)
        """
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_images(self, tag, images, step=None, nrow=8, normalize=True, value_range=(-1, 1)):
        """
        Log images to TensorBoard
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            tag: Tag name
            images: Tensor of images (N, C, H, W)
            step: Step number (uses global_step if None)
            nrow: Number of images per row in grid
            normalize: Whether to normalize images
            value_range: Range for normalization
        """
        if step is None:
            step = self.global_step
        
        # Create grid
        grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=value_range)
        self.writer.add_image(tag, grid, step)
    
    def log_model_graph(self, model, input_shape):
        """
        Log model graph
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            model: Model to log
            input_shape: Input tensor shape
        """
        dummy_input = torch.randn(input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def log_histogram(self, tag, values, step=None):
        """
        Log histogram
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            tag: Tag name
            values: Values to create histogram from
            step: Step number (uses global_step if None)
        """
        if step is None:
            step = self.global_step
        self.writer.add_histogram(tag, values, step)
    
    def log_losses(self, g_loss, d_loss, step=None):
        """
        Log generator and discriminator losses
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            g_loss: Generator loss
            d_loss: Discriminator loss
            step: Step number (uses global_step if None)
        """
        self.log_scalar('Loss/Generator', g_loss, step)
        self.log_scalar('Loss/Discriminator', d_loss, step)
        self.log_scalar('Loss/Total', g_loss + d_loss, step)
    
    def log_learning_rates(self, lr_g, lr_d, step=None):
        """
        Log learning rates
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
            step: Step number (uses global_step if None)
        """
        self.log_scalar('LearningRate/Generator', lr_g, step)
        self.log_scalar('LearningRate/Discriminator', lr_d, step)
    
    def increment_step(self):
        """Increment global step counter"""
        self.global_step += 1
    
    def close(self):
        """Close the writer"""
        self.writer.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

