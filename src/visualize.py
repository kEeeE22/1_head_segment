"""
Visualization functions for images and masks.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_sample(sample, figsize=(18, 6)):
    """
    Visualize image and mask from a sample dictionary.
    
    Args:
        sample (dict): Dictionary with 'image' and 'wall_mask' keys
        figsize (tuple): Figure size for matplotlib
    """
    img_tensor = sample['image']      # Shape: (3, H, W)
    mask_tensor = sample['wall_mask'] # Shape: (H, W)

    # Convert to numpy if needed
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = img_tensor.copy()

    # Transpose if channels first
    if img.shape[0] == 3: 
        img = np.transpose(img, (1, 2, 0))

    # Normalize to [0, 1] for display
    img = img - img.min() 
    if img.max() > 0:
        img = img / img.max()

    # Convert mask to numpy
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.detach().cpu().numpy()
    else:
        mask = mask_tensor.copy()
        
    mask = mask.squeeze() 

    # Create figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # 1. Original image
    ax[0].imshow(img)
    ax[0].set_title("Original Image (Normalized)")
    ax[0].axis('off')

    # 2. Wall mask
    ax[1].imshow(mask, cmap='gray', interpolation='nearest') 
    ax[1].set_title(f"Wall Mask (Unique values: {np.unique(mask)})")
    ax[1].axis('off')

    # 3. Overlay
    ax[2].imshow(img)
    
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
    overlay[mask > 0] = [1, 0, 0, 0.5]  # Red with 50% transparency

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay Check")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history metrics.
    
    Args:
        history (dict): Training history from train_full_metrics
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train IoU')
    axes[0, 1].plot(history['val_iou'], label='Val IoU')
    axes[0, 1].set_title('IoU Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision & Recall
    axes[1, 1].plot(history['train_prec'], label='Train Precision')
    axes[1, 1].plot(history['val_prec'], label='Val Precision')
    axes[1, 1].plot(history['train_rec'], label='Train Recall')
    axes[1, 1].plot(history['val_rec'], label='Val Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()
