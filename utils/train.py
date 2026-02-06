"""
Training loop and metrics calculation.
"""

import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

from config import DEVICE


def train_full_metrics(model, trainloader, valloader, criterion, optimizer, 
                       epochs, device=DEVICE, save_path='best_model.pth'):
    """
    Train model with full metrics tracking.
    
    Args:
        model (torch.nn.Module): Model to train
        trainloader (DataLoader): Training data loader
        valloader (DataLoader): Validation data loader
        criterion (callable): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        epochs (int): Number of epochs to train
        device (str): Device to train on
        save_path (str): Path to save best model
        
    Returns:
        dict: Training history with losses and metrics
    """
    best_iou = 0.0
    
    # Dictionary to store training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [],  'val_f1': [],
        'train_prec': [], 'val_prec': [],
        'train_rec': [], 'val_rec': []
    }

    for epoch in range(epochs):
        # --- 1. TRAINING PHASE ---
        model.train()
        train_loss = 0
        
        # Metrics accumulators
        metrics = {'iou': 0, 'f1': 0, 'prec': 0, 'rec': 0}
        
        loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{epochs}] - Train")
        
        for samples in loop:
            images = samples['image'].to(device)
            masks = samples['wall_mask'].to(device)
            
            # Forward & Backward
            logits = model(images)
            loss = criterion(logits, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # --- CALCULATE METRICS ---
            preds = torch.argmax(logits, dim=1)
            
            # Get statistics for metrics calculation
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds, masks.long(), 
                mode='multiclass', num_classes=3
            )
            
            # Calculate all metrics from stats
            metrics['iou'] += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            metrics['f1']  += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            metrics['prec']+= smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
            metrics['rec'] += smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

            # Update progress bar
            loop.set_postfix(loss=loss.item(), iou=metrics['iou']/(loop.n+1))
        
        # Calculate epoch averages for training
        n_train = len(trainloader)
        avg_train_loss = train_loss / n_train
        avg_train_iou = metrics['iou'] / n_train
        avg_train_f1  = metrics['f1'] / n_train
        avg_train_prec= metrics['prec'] / n_train
        avg_train_rec = metrics['rec'] / n_train

        # --- 2. VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_metrics = {'iou': 0, 'f1': 0, 'prec': 0, 'rec': 0}
        
        with torch.no_grad():
            for samples in valloader:
                images = samples['image'].to(device)
                masks = samples['wall_mask'].to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='multiclass', num_classes=3)
                
                val_metrics['iou'] += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
                val_metrics['f1']  += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
                val_metrics['prec']+= smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
                val_metrics['rec'] += smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
            
        n_val = len(valloader)
        avg_val_loss = val_loss / n_val
        avg_val_iou = val_metrics['iou'] / n_val
        avg_val_f1  = val_metrics['f1'] / n_val
        avg_val_prec= val_metrics['prec'] / n_val
        avg_val_rec = val_metrics['rec'] / n_val

        # --- 3. PRINT RESULTS ---
        print(f"\nEpoch [{epoch+1}/{epochs}] Result:")
        print(f"Train | Loss:{avg_train_loss:.3f} | IoU:{avg_train_iou:.3f} | F1:{avg_train_f1:.3f} | Pre:{avg_train_prec:.3f} | Rec:{avg_train_rec:.3f}")
        print(f"Val   | Loss:{avg_val_loss:.3f} | IoU:{avg_val_iou:.3f} | F1:{avg_val_f1:.3f} | Pre:{avg_val_prec:.3f} | Rec:{avg_val_rec:.3f}")
        
        # Save to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        history['train_f1'].append(avg_train_f1)
        history['val_f1'].append(avg_val_f1)
        history['train_prec'].append(avg_train_prec)
        history['val_prec'].append(avg_val_prec)
        history['train_rec'].append(avg_train_rec)
        history['val_rec'].append(avg_val_rec)

        # Save best model
        if avg_val_iou > best_iou:
            print(f"--> IoU improves to {avg_val_iou:.4f}. Saving...")
            best_iou = avg_val_iou
            torch.save(model.state_dict(), save_path)
            
    return history
