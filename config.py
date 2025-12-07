"""
Configuration file for DCGAN training

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This file contains all hyperparameters and configuration settings
for the DCGAN training process.
"""

# Model hyperparameters
NZ = 100  # Size of input noise vector
NGF = 64  # Number of generator filters in first conv layer
NDF = 64  # Number of discriminator filters in first conv layer
NC = 3    # Number of channels in training images (3 for RGB, 1 for grayscale)

# Training hyperparameters
BATCH_SIZE = 128
IMAGE_SIZE = 64  # Size of images (64x64 or 128x128)
NUM_EPOCHS = 50
LR = 0.0002  # Learning rate for optimizers
BETA1 = 0.5  # Beta1 hyperparameter for Adam optimizers

# Dataset configuration
DATASET_NAME = 'custom'  # 'custom', 'celeba', 'cifar10', 'mnist'
DATA_ROOT = './data'
CUSTOM_DATA_DIR = './data/custom'  # Required if DATASET_NAME='custom'
NUM_WORKERS = 2

# Training settings
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

# Output settings
OUTPUT_DIR = './outputs'
SAVE_IMAGES_INTERVAL = 100  # Save generated images every N iterations
PLOT_LOSSES = True

# Visualization settings
NUM_VISUALIZATION_SAMPLES = 64

