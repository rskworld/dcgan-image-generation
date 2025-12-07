# Advanced Features Documentation

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

This document describes all the advanced features available in the DCGAN project.

## Table of Contents

1. [Evaluation Metrics](#evaluation-metrics)
2. [Advanced Training Features](#advanced-training-features)
3. [TensorBoard Integration](#tensorboard-integration)
4. [Latent Space Interpolation](#latent-space-interpolation)
5. [Data Augmentation](#data-augmentation)
6. [Resume Training](#resume-training)
7. [Web Interface](#web-interface)

## Evaluation Metrics

### FID (Fréchet Inception Distance)
Measures the quality and diversity of generated images by comparing feature distributions.

**Usage:**
```python
from evaluation import FIDScore

fid_calculator = FIDScore(device='cuda')
fid_score = fid_calculator.calculate(real_images, fake_images)
```

**Lower FID = Better quality**

### IS (Inception Score)
Measures the quality and diversity of generated images.

**Usage:**
```python
from evaluation import InceptionScore

is_calculator = InceptionScore(device='cuda')
is_score, is_std = is_calculator.calculate(fake_images)
```

**Higher IS = Better quality**

### Complete Evaluation
```bash
python scripts/evaluate_model.py --checkpoint checkpoints/final_generator.pth
```

## Advanced Training Features

### Learning Rate Scheduling
Adjust learning rate during training for better convergence.

**Available schedulers:**
- Linear decay
- Cosine annealing
- Step decay
- Reduce on plateau

**Usage:**
```python
from training_utils import LearningRateScheduler

scheduler = LearningRateScheduler(
    optimizer,
    scheduler_type='cosine',
    start_lr=0.0002,
    end_lr=0.00002,
    total_steps=1000
)
scheduler.step()
```

### Gradient Clipping
Prevent exploding gradients.

**Usage:**
```python
from training_utils import GradientClipper

GradientClipper.clip_gradients(model, max_norm=1.0)
```

### Label Smoothing
Improve training stability.

**Usage:**
```python
from training_utils import LabelSmoothing

smooth_labels = LabelSmoothing.smooth_labels(labels, smoothing=0.1)
```

### Early Stopping
Stop training when model stops improving.

**Usage:**
```python
from training_utils import EarlyStopping

early_stopping = EarlyStopping(patience=10, mode='min')
if early_stopping(metric_score):
    break  # Stop training
```

## TensorBoard Integration

Visualize training progress in real-time.

**Usage:**
```python
from tensorboard_logger import TensorBoardLogger

with TensorBoardLogger(log_dir='runs/dcgan') as logger:
    logger.log_losses(g_loss, d_loss)
    logger.log_images('Generated', fake_images)
    logger.increment_step()
```

**View results:**
```bash
tensorboard --logdir runs/dcgan
```

Then open http://localhost:6006 in your browser.

## Latent Space Interpolation

Generate smooth transitions between images by interpolating in latent space.

### Linear Interpolation
```python
from latent_interpolation import visualize_interpolation

z1 = torch.randn(1, 100, 1, 1)
z2 = torch.randn(1, 100, 1, 1)

visualize_interpolation(
    generator, z1, z2,
    num_steps=10,
    interpolation_type='linear',
    save_path='outputs/interpolation.png'
)
```

### Spherical Interpolation (SLERP)
```python
visualize_interpolation(
    generator, z1, z2,
    num_steps=10,
    interpolation_type='spherical',
    save_path='outputs/interpolation.png'
)
```

### Latent Walk
```python
from latent_interpolation import generate_latent_walk

generate_latent_walk(
    generator,
    nz=100,
    num_images=20,
    save_path='outputs/latent_walk.png'
)
```

**Command line:**
```bash
python scripts/interpolate_latent.py --checkpoint checkpoints/final_generator.pth
```

## Data Augmentation

### Standard Augmentation
```python
from data_augmentation import AdvancedAugmentation

transform = AdvancedAugmentation.get_augmentation_transform(
    image_size=64,
    augmentation_level='medium'  # 'light', 'medium', 'heavy'
)
```

### Adaptive Augmentation
Automatically adjusts augmentation based on discriminator performance.

```python
from data_augmentation import AdaptiveAugmentation

ada_aug = AdaptiveAugmentation(initial_prob=0.0, target_prob=0.6)
ada_aug.update(real_score, fake_score)

if ada_aug.should_augment():
    # Apply augmentation
    pass
```

### Mixup
```python
from data_augmentation import AdvancedAugmentation

augmented_images = AdvancedAugmentation.mixup(images, alpha=0.2)
```

### Cutout
```python
from data_augmentation import AdvancedAugmentation

augmented_images = AdvancedAugmentation.cutout(images, num_holes=1, length=16)
```

## Resume Training

Resume training from the latest checkpoint.

**Usage:**
```python
from resume_training import resume_training

start_epoch = resume_training('checkpoints', trainer)
```

**Command line:**
```bash
python resume_training_script.py
```

## Web Interface

Launch a web interface for generating images.

**Start server:**
```bash
python web_app.py
```

Then open http://localhost:5000 in your browser.

**Features:**
- Generate images interactively
- Adjust number of images
- Real-time generation
- Download generated images

## Example: Complete Training with Advanced Features

```python
from trainer import DCGANTrainer
from training_utils import LearningRateScheduler, GradientClipper, EarlyStopping
from tensorboard_logger import TensorBoardLogger
from evaluation import evaluate_model

# Create trainer
trainer = DCGANTrainer(...)

# Setup learning rate scheduler
lr_scheduler = LearningRateScheduler(
    trainer.optimizerG,
    scheduler_type='cosine',
    total_steps=1000
)

# Setup early stopping
early_stopping = EarlyStopping(patience=10)

# Setup TensorBoard
with TensorBoardLogger() as logger:
    for epoch in range(num_epochs):
        # Training...
        
        # Clip gradients
        GradientClipper.clip_gradients(trainer.netG, max_norm=1.0)
        GradientClipper.clip_gradients(trainer.netD, max_norm=1.0)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Log to TensorBoard
        logger.log_losses(g_loss, d_loss)
        logger.log_images('Generated', fake_images)
        
        # Check early stopping
        if early_stopping(g_loss):
            break

# Evaluate model
results = evaluate_model(trainer.netG, dataloader, device)
print(f"FID: {results['fid_score']:.4f}")
print(f"IS: {results['inception_score']:.4f}")
```

## Contact

For questions or support:
- **Website:** https://rskworld.in
- **Email:** help@rskworld.in
- **Phone:** +91 93305 39277

