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
        targets (torch.Tensor): Target one-hot encoded masks of shape (B, C, H, W)
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    eps = 1e-6
    m = nn.Softmax(dim=1)
    z = m(preds)
    z = torch.clamp(z, eps, 1 - eps)
    log_z = torch.log(z)

    num_classes = targets.size(1)
    ind = torch.argmax(targets, 1).long()
    total = torch.sum(targets)

    m_c, n_c = [], []
    for c in range(num_classes):
        m_c.append((ind == c).int())
        n_c.append(torch.sum(m_c[-1]).float())

    c = [total - n_c[i] for i in range(num_classes)]
    tc = sum(c)

    loss = 0
    for i in range(num_classes):
        w = c[i] / tc
        m_c_one_hot = F.one_hot(
            (i * m_c[i]).permute(1, 2, 0).long(),
            num_classes
        ).permute(2, 3, 0, 1)

        y_c = m_c_one_hot * targets
        loss += w * torch.sum(-torch.sum(y_c * log_z, dim=2))

    return loss / num_classes


def cross_two_tasks_weight(rooms, boundaries):
    """
    Calculate dynamic weights for balancing two tasks based on pixel counts.
    
    Args:
        rooms (torch.Tensor): Room segmentation tensor
        boundaries (torch.Tensor): Boundary segmentation tensor
        
    Returns:
        tuple: (w1, w2) weights for room and boundary tasks
    """
    p1 = torch.sum(rooms).float()
    p2 = torch.sum(boundaries).float()
    w1 = p2 / (p1 + p2)
    w2 = p1 / (p1 + p2)
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
