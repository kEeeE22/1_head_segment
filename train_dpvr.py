import argparse
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp

from data_class import DPVR_cubicasa, get_train_transform, get_val_transform
# =========================
# TensorBoard
# =========================
writer = SummaryWriter('log/store2')

# =========================
# Loss functions
# =========================
def balanced_entropy(preds, targets):
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
    p1 = torch.sum(rooms).float()
    p2 = torch.sum(boundaries).float()
    w1 = p2 / (p1 + p2)
    w2 = p1 / (p1 + p2)
    return w1, w2

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


def getloader(args):
    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = DPVR_cubicasa('/dataset/cubicasa5k/cubicasa5k/', 'train.txt', transform=get_train_transform)
    val_dataset = DPVR_cubicasa('/dataset/cubicasa5k/cubicasa5k/', 'val.txt', transform=get_val_transform)

    print(len(train_dataset))
    print(len(val_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.numworkers,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.numworkers
    )

    return train_loader, val_loader

# =========================
# Main train loop
# =========================
def main(args):
    device, model, optimizer = setup(args)
    train_loader, val_loader = getloader(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = f"log/store2/checkpoint_{timestamp}.pt"

    if args.earlystop:
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.maxiters):
        # ================= TRAIN =================
        model.train()
        train_loss = 0

        train_r = {'iou':0, 'f1':0, 'prec':0, 'rec':0}
        train_cw = {'iou':0, 'f1':0, 'prec':0, 'rec':0}

        for im, cw, r, _ in train_loader:
            im, cw, r = im.to(device), cw.to(device), r.to(device)
            optimizer.zero_grad()

            logits_r, logits_cw = model(im)

            loss_r = balanced_entropy(logits_r, r)
            loss_cw = balanced_entropy(logits_cw, cw)
            w1, w2 = cross_two_tasks_weight(r, cw)
            loss = w1 * loss_r + w2 * loss_cw

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            mr = compute_metrics(logits_r, r, r.size(1))
            mcw = compute_metrics(logits_cw, cw, cw.size(1))

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
            for im, cw, r, _ in val_loader:
                im, cw, r = im.to(device), cw.to(device), r.to(device)

                logits_r, logits_cw = model(im)

                loss_r = balanced_entropy(logits_r, r)
                loss_cw = balanced_entropy(logits_cw, cw)
                w1, w2 = cross_two_tasks_weight(r, cw)
                loss = w1 * loss_r + w2 * loss_cw

                val_loss += loss.item()

                mr = compute_metrics(logits_r, r, r.size(1))
                mcw = compute_metrics(logits_cw, cw, cw.size(1))

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

        if args.earlystop:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            torch.save(model.state_dict(), ckpt_path)

    torch.save(model.state_dict(), ckpt_path)
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
