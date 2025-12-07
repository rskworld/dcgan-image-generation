"""
Evaluation Metrics for DCGAN

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides evaluation metrics for GAN models including:
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Visual quality metrics
"""

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
import os


class InceptionScore:
    """
    Calculate Inception Score (IS) for generated images
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, device='cuda', batch_size=32):
        """
        Initialize Inception Score calculator
        
        Args:
            device: torch device
            batch_size: Batch size for evaluation
        """
        self.device = device
        self.batch_size = batch_size
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        self.model.fc = torch.nn.Identity()  # Remove final classification layer
    
    def calculate(self, images):
        """
        Calculate Inception Score
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            images: Tensor of images (N, C, H, W) in range [0, 1]
        
        Returns:
            IS score (mean, std)
        """
        # Resize images to 299x299 for Inception v3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] for Inception v3
        images = (images - 0.5) * 2
        
        preds = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size].to(self.device)
                pred = self.model(batch)
                preds.append(F.softmax(pred, dim=1).cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        py = np.mean(preds, axis=0)
        scores = []
        for i in range(preds.shape[0]):
            pyx = preds[i]
            scores.append(np.sum(pyx * np.log(pyx / py)))
        
        is_score = np.exp(np.mean(scores))
        is_std = np.exp(np.std(scores))
        
        return is_score, is_std


class FIDScore:
    """
    Calculate Fréchet Inception Distance (FID) for generated images
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, device='cuda', batch_size=32):
        """
        Initialize FID calculator
        
        Args:
            device: torch device
            batch_size: Batch size for evaluation
        """
        self.device = device
        self.batch_size = batch_size
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        # Use the last pooling layer before classification
        self.model.fc = torch.nn.Identity()
    
    def get_features(self, images):
        """
        Extract features from images using Inception v3
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            images: Tensor of images (N, C, H, W) in range [0, 1]
        
        Returns:
            Feature vectors
        """
        # Resize images to 299x299 for Inception v3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] for Inception v3
        images = (images - 0.5) * 2
        
        features = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size].to(self.device)
                feat = self.model(batch)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate Fréchet distance between two multivariate Gaussians
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % 1e-6
            print(msg)
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    
    def calculate(self, real_images, fake_images):
        """
        Calculate FID score
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            real_images: Tensor of real images (N, C, H, W) in range [0, 1]
            fake_images: Tensor of fake images (N, C, H, W) in range [0, 1]
        
        Returns:
            FID score
        """
        # Get features
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # Calculate statistics
        mu1 = np.mean(real_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        
        mu2 = np.mean(fake_features, axis=0)
        sigma2 = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        fid = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return fid


def evaluate_model(generator, real_dataloader, device, num_samples=5000):
    """
    Evaluate generator model using IS and FID
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        generator: Trained generator model
        real_dataloader: DataLoader with real images
        device: torch device
        num_samples: Number of samples for evaluation
    
    Returns:
        Dictionary with IS and FID scores
    """
    generator.eval()
    
    # Collect real images
    real_images = []
    for batch, _ in real_dataloader:
        real_images.append(batch)
        if len(real_images) * real_images[0].shape[0] >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    # Denormalize from [-1, 1] to [0, 1]
    real_images = (real_images + 1) / 2
    
    # Generate fake images
    nz = 100
    fake_images = []
    with torch.no_grad():
        for _ in range(0, num_samples, 64):
            noise = torch.randn(min(64, num_samples - len(fake_images)), nz, 1, 1, device=device)
            fake = generator(noise).cpu()
            fake_images.append(fake)
    
    fake_images = torch.cat(fake_images, dim=0)[:num_samples]
    # Denormalize from [-1, 1] to [0, 1]
    fake_images = (fake_images + 1) / 2
    
    # Calculate metrics
    print("Calculating Inception Score...")
    is_calculator = InceptionScore(device=device)
    is_score, is_std = is_calculator.calculate(fake_images)
    
    print("Calculating FID Score...")
    fid_calculator = FIDScore(device=device)
    fid_score = fid_calculator.calculate(real_images, fake_images)
    
    results = {
        'inception_score': is_score,
        'inception_score_std': is_std,
        'fid_score': fid_score
    }
    
    generator.train()
    
    return results

