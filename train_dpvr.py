import argparse
import random
import os
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
from net import DFPmodel
from data_class import DPVR_cubicasa, get_train_transform, get_val_transform
from utils.losses import balanced_entropy, cross_two_tasks_weight
from config import LOG_DIR, DEFAULT_DATA_FOLDER

# TensorBoard
writer = SummaryWriter(LOG_DIR)


# =========================
# Metrics helper
# =========================
def compute_metrics(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)
    gt = torch.argmax(targets, dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, gt,
        mode="multiclass",
        num_classes=num_classes
    )

    return {
        "iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise"),
        "f1": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise"),
        "prec": smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise"),
        "rec": smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise"),
    }

# =========================
# Setup
# =========================
def setup(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFPmodel(pretrained=args.pretrained, freeze=args.freeze)

    if args.loadmodel:
        model.load_state_dict(torch.load(args.loadmodel))

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    return device, model, optimizer


def get_loader(args):
    """Create train and validation data loaders."""
    train_dataset = DPVR_cubicasa(
        DEFAULT_DATA_FOLDER, 
        'train.txt', 
        transform=get_train_transform()
    )
    val_dataset = DPVR_cubicasa(
        DEFAULT_DATA_FOLDER, 
        'val.txt', 
        transform=get_val_transform()
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.numworkers,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.numworkers,
        shuffle=False
    )

    return train_loader, val_loader


def main(args):
    """Main training function."""
    device, model, optimizer = setup(args)
    train_loader, val_loader = get_loader(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(LOG_DIR, f"checkpoint_{timestamp}.pt")
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.maxiters):
        # ================= TRAIN =================
        model.train()
        train_loss = 0

        train_r = {'iou':0, 'f1':0, 'prec':0, 'rec':0}
        train_cw = {'iou':0, 'f1':0, 'prec':0, 'rec':0}

        for batch in train_loader:
            im = batch['image'].to(device)
            cw = batch['boundary'].to(device)
            r = batch['room'].to(device)
            
            optimizer.zero_grad()

            logits_r, logits_cw = model(im)

            # Loss function now accepts class indices directly
            loss_r = balanced_entropy(logits_r, r)
            loss_cw = balanced_entropy(logits_cw, cw)
            
            # For task weighting, use pixel counts
            w1 = torch.sum(r > 0).float()
            w2 = torch.sum(cw > 0).float()
            total = w1 + w2
            w1, w2 = w2 / total, w1 / total  # Inverse weighting
            
            loss = w1 * loss_r + w2 * loss_cw

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # For metrics, convert predictions to class indices
            mr = compute_metrics(logits_r, F.one_hot(r.long(), num_classes=logits_r.size(1)).permute(0, 3, 1, 2), logits_r.size(1))
            mcw = compute_metrics(logits_cw, F.one_hot(cw.long(), num_classes=logits_cw.size(1)).permute(0, 3, 1, 2), logits_cw.size(1))

            for k in train_r:
                train_r[k] += mr[k]
                train_cw[k] += mcw[k]

        n_train = len(train_loader)
        for k in train_r:
            train_r[k] /= n_train
            train_cw[k] /= n_train

        # ================= VALID =================
        model.eval()
        val_loss = 0

        val_r = {'iou':0, 'f1':0, 'prec':0, 'rec':0}
        val_cw = {'iou':0, 'f1':0, 'prec':0, 'rec':0}

        with torch.no_grad():
            for batch in val_loader:
                im = batch['image'].to(device)
                cw = batch['boundary'].to(device)
                r = batch['room'].to(device)

                logits_r, logits_cw = model(im)

                # Loss function now accepts class indices directly
                loss_r = balanced_entropy(logits_r, r)
                loss_cw = balanced_entropy(logits_cw, cw)
                
                # For task weighting, use pixel counts
                w1 = torch.sum(r > 0).float()
                w2 = torch.sum(cw > 0).float()
                total = w1 + w2
                w1, w2 = w2 / total, w1 / total
                
                loss = w1 * loss_r + w2 * loss_cw

                val_loss += loss.item()

                # For metrics, convert to one-hot
                mr = compute_metrics(logits_r, F.one_hot(r.long(), num_classes=logits_r.size(1)).permute(0, 3, 1, 2), logits_r.size(1))
                mcw = compute_metrics(logits_cw, F.one_hot(cw.long(), num_classes=logits_cw.size(1)).permute(0, 3, 1, 2), logits_cw.size(1))

                for k in val_r:
                    val_r[k] += mr[k]
                    val_cw[k] += mcw[k]

        n_val = len(val_loader)
        for k in val_r:
            val_r[k] /= n_val
            val_cw[k] /= n_val

        # ================= LOG =================
        print(
            f"\nEpoch {epoch} | "
            f"Train IoU (R/CW): {train_r['iou']:.3f}/{train_cw['iou']:.3f} | "
            f"Val IoU (R/CW): {val_r['iou']:.3f}/{val_cw['iou']:.3f}"
        )

        writer.add_scalars('IoU/train', {
            'room': train_r['iou'],
            'boundary': train_cw['iou']
        }, epoch)

        writer.add_scalars('IoU/val', {
            'room': val_r['iou'],
            'boundary': val_cw['iou']
        }, epoch)

        # Save best model
        if args.earlystop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nTraining completed. Final model saved to {ckpt_path}")
    return model

# =========================
# Entry
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--maxiters', type=int, default=2000)
    p.add_argument('--numworkers', type=int, default=1)
    p.add_argument('--pretrained', type=bool, default=True)
    p.add_argument('--freeze', type=bool, default=True)
    p.add_argument('--loadmodel', type=str, default=None)
    p.add_argument('--valsplit', type=float, default=0.1)
    p.add_argument('--tensorboard', type=bool, default=True)
    p.add_argument('--earlystop', type=bool, default=False)
    p.add_argument('--patience', type=int, default=20)

    args = p.parse_args()
    main(args)
