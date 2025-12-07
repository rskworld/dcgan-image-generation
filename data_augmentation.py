"""
Data Augmentation Utilities for DCGAN

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides advanced data augmentation techniques for GAN training.
"""

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class AdvancedAugmentation:
    """
    Advanced data augmentation pipeline
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    @staticmethod
    def get_augmentation_transform(image_size=64, augmentation_level='medium'):
        """
        Get augmentation transform based on level
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            image_size: Target image size
            augmentation_level: 'light', 'medium', or 'heavy'
        
        Returns:
            Transform composition
        """
        if augmentation_level == 'light':
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif augmentation_level == 'medium':
            return transforms.Compose([
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:  # heavy
            return transforms.Compose([
                transforms.Resize(int(image_size * 1.2)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
            ])
    
    @staticmethod
    def mixup(images, alpha=0.2):
        """
        Mixup augmentation
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            images: Batch of images
            alpha: Mixup parameter
        
        Returns:
            Mixed images
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        return mixed_images
    
    @staticmethod
    def cutout(images, num_holes=1, length=16):
        """
        Cutout augmentation
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            images: Batch of images
            num_holes: Number of holes to cut
            length: Length of each hole
        
        Returns:
            Images with cutout
        """
        h = images.size(2)
        w = images.size(3)
        
        for n in range(images.size(0)):
            for _ in range(num_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                
                images[n, :, y1:y2, x1:x2] = 0
        
        return images


class AdaptiveAugmentation:
    """
    Adaptive augmentation that adjusts based on discriminator performance
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, initial_prob=0.0, target_prob=0.6):
        """
        Initialize adaptive augmentation
        
        Args:
            initial_prob: Initial augmentation probability
            target_prob: Target augmentation probability
        """
        self.prob = initial_prob
        self.target_prob = target_prob
        self.step_size = 0.01
    
    def update(self, real_score, fake_score):
        """
        Update augmentation probability based on discriminator performance
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            real_score: Discriminator score on real images
            fake_score: Discriminator score on fake images
        """
        # If discriminator is too strong (high real score, low fake score)
        # increase augmentation
        if real_score > 0.8 and fake_score < 0.2:
            self.prob = min(self.prob + self.step_size, self.target_prob)
        # If discriminator is balanced, decrease augmentation
        elif 0.3 < real_score < 0.7 and 0.3 < fake_score < 0.7:
            self.prob = max(self.prob - self.step_size, 0.0)
    
    def should_augment(self):
        """Check if augmentation should be applied"""
        return np.random.random() < self.prob
    
    def get_prob(self):
        """Get current augmentation probability"""
        return self.prob

