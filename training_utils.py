"""
Advanced Training Utilities for DCGAN

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides advanced training utilities including:
- Learning rate scheduling
- Gradient clipping
- Label smoothing
- Early stopping
- Training callbacks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, List


class LearningRateScheduler:
    """
    Learning rate scheduler for GAN training
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, optimizer, scheduler_type='linear', **kwargs):
        """
        Initialize learning rate scheduler
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler ('linear', 'cosine', 'step', 'plateau')
            **kwargs: Additional scheduler parameters
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
        self.step_count = 0
        
        if scheduler_type == 'linear':
            self.start_lr = kwargs.get('start_lr', optimizer.param_groups[0]['lr'])
            self.end_lr = kwargs.get('end_lr', self.start_lr * 0.1)
            self.total_steps = kwargs.get('total_steps', 1000)
        elif scheduler_type == 'cosine':
            self.start_lr = kwargs.get('start_lr', optimizer.param_groups[0]['lr'])
            self.end_lr = kwargs.get('end_lr', self.start_lr * 0.01)
            self.total_steps = kwargs.get('total_steps', 1000)
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10)
            )
    
    def step(self, metric=None):
        """
        Update learning rate
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        """
        self.step_count += 1
        
        if self.scheduler_type == 'linear':
            lr = self.start_lr - (self.start_lr - self.end_lr) * (self.step_count / self.total_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(lr, self.end_lr)
        elif self.scheduler_type == 'cosine':
            lr = self.end_lr + (self.start_lr - self.end_lr) * 0.5 * (1 + np.cos(np.pi * self.step_count / self.total_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(lr, self.end_lr)
        elif self.scheduler_type == 'step':
            self.scheduler.step()
        elif self.scheduler_type == 'plateau':
            if metric is not None:
                self.scheduler.step(metric)
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class GradientClipper:
    """
    Gradient clipping utility
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    @staticmethod
    def clip_gradients(model, max_norm=1.0):
        """
        Clip gradients by norm
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            model: Model to clip gradients for
            max_norm: Maximum gradient norm
        
        Returns:
            Total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm.item()


class LabelSmoothing:
    """
    Label smoothing for GAN training
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    @staticmethod
    def smooth_labels(tensor, smoothing=0.1):
        """
        Apply label smoothing
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            tensor: Target labels
            smoothing: Smoothing factor (0.0 to 1.0)
        
        Returns:
            Smoothed labels
        """
        return tensor * (1.0 - smoothing) + 0.5 * smoothing


class EarlyStopping:
    """
    Early stopping callback
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        """
        Check if training should stop
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        
        Args:
            score: Current metric score
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class TrainingCallback:
    """
    Base class for training callbacks
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def on_epoch_start(self, epoch):
        """Called at the start of each epoch"""
        pass
    
    def on_epoch_end(self, epoch, losses):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_end(self, batch, losses):
        """Called at the end of each batch"""
        pass


class MetricsTracker:
    """
    Track training metrics
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics = {}
    
    def update(self, key, value):
        """
        Update metric
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        """
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def get_mean(self, key, window=None):
        """
        Get mean of metric
        
        Author: RSK World
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        """
        if key not in self.metrics:
            return None
        values = self.metrics[key]
        if window:
            values = values[-window:]
        return np.mean(values)
    
    def get_latest(self, key):
        """Get latest value of metric"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None
        return self.metrics[key][-1]

