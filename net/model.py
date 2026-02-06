"""
Model initialization and loss functions.
"""

import torch
import segmentation_models_pytorch as smp

from .config import (
    DEVICE, 
    ENCODER_NAME, 
    ENCODER_WEIGHTS, 
    NUM_CLASSES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY
)


def create_model(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, 
                 num_classes=NUM_CLASSES, device=DEVICE):
    """
    Create UNet++ model for wall segmentation.
    
    Args:
        encoder_name (str): Encoder backbone name
        encoder_weights (str): Pretrained weights ('imagenet' or None)
        num_classes (int): Number of output classes
        device (str): Device to move model to
        
    Returns:
        torch.nn.Module: Initialized model
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    model.to(device)
    return model


def get_criterion():
    """
    Get combined loss function (Dice + Focal).
    
    Returns:
        callable: Loss function
    """
    dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=None)
    
    def criterion(y_pred, y_true):
        return 0.5 * dice_loss(y_pred, y_true) + 0.5 * focal_loss(y_pred, y_true)
    
    return criterion


def get_optimizer(model, lr=DEFAULT_LEARNING_RATE, weight_decay=DEFAULT_WEIGHT_DECAY):
    """
    Get AdamW optimizer.
    
    Args:
        model (torch.nn.Module): Model to optimize
        lr (float): Learning rate
        weight_decay (float): Weight decay coefficient
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
