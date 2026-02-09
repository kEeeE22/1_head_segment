"""
Loss functions for DPVR dual-task segmentation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def balanced_entropy(preds, targets):
    """
    Balanced cross-entropy loss for handling class imbalance.
    
    Args:
        preds (torch.Tensor): Predicted logits of shape (B, C, H, W)
        targets (torch.Tensor): Target class indices of shape (B, H, W)
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    eps = 1e-6
    B, C, H, W = preds.shape
    
    # Apply softmax to get probabilities
    probs = F.softmax(preds, dim=1)
    probs = torch.clamp(probs, eps, 1 - eps)
    log_probs = torch.log(probs)
    
    # Ensure targets are long type
    targets = targets.long()
    
    # Calculate class weights based on frequency (inverse frequency weighting)
    total_pixels = B * H * W
    class_weights = []
    
    for c in range(C):
        n_c = torch.sum((targets == c).float())
        # Weight is inversely proportional to class frequency
        weight = (total_pixels - n_c) / total_pixels if n_c > 0 else 1.0
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, device=preds.device)
    
    # Compute weighted cross-entropy
    # Gather log probabilities for ground truth classes
    targets_flat = targets.view(B, 1, H, W)
    log_probs_gt = torch.gather(log_probs, 1, targets_flat).squeeze(1)
    
    # Apply class weights
    weights_map = class_weights[targets]
    
    # Compute loss
    loss = -torch.sum(weights_map * log_probs_gt) / total_pixels
    
    return loss



def cross_two_tasks_weight(rooms, boundaries):
    """
    Calculate dynamic weights based on pixel counts (Indices version).
    Matches logic of main.py but adapted for Index inputs.
    """
    # SỬA: Đếm số pixel khác 0 (foreground) thay vì tính tổng giá trị
    p1 = torch.sum(rooms > 0).float()
    p2 = torch.sum(boundaries > 0).float()
    
    # Tránh lỗi chia cho 0
    total = p1 + p2 + 1e-6
    
    # Logic đảo nghịch trọng số giống main.py:
    # Task nào ít diện tích hơn (p nhỏ) sẽ được gán trọng số lớn hơn
    # main.py: w1 (room weight) = p2 (boundary pixels) / total
    w1 = p2 / total
    w2 = p1 / total
    
    return w1, w2


def get_dpvr_criterion():
    """
    Get combined criterion for DPVR dual-task model.
    
    Returns:
        callable: Loss function that takes (logit_r, logit_b, r, b)
    """
    def criterion(logit_r, logit_b, r, b):
        """
        Args:
            logit_r: Room logits
            logit_b: Boundary logits
            r: Room targets
            b: Boundary targets
        """
        w1, w2 = cross_two_tasks_weight(r, b)
        return w1 * balanced_entropy(logit_r, r) + w2 * balanced_entropy(logit_b, b)
    
    return criterion
