"""
Main training script for CubiCasa wall segmentation.

Usage:
    python train_model.py --data_folder /path/to/data --epochs 100
"""

import argparse
import os
from torch.utils.data import DataLoader

from src import (
    FloorplanSVG,
    get_train_transform,
    get_val_transform,
    create_model,
    get_criterion,
    get_optimizer,
    train_full_metrics,
    visualize_sample,
    DEVICE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_EPOCHS,
    DEFAULT_DATA_FOLDER,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
)


def main():
    parser = argparse.ArgumentParser(description='Train CubiCasa wall segmentation model')
    
    # Data arguments
    parser.add_argument('--data_folder', type=str, default=DEFAULT_DATA_FOLDER,
                        help='Root folder containing the CubiCasa5k dataset')
    parser.add_argument('--train_file', type=str, default=TRAIN_FILE,
                        help='Training data list file')
    parser.add_argument('--val_file', type=str, default=VAL_FILE,
                        help='Validation data list file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone (resnet34, resnet50, etc.)')
    parser.add_argument('--save_path', type=str, default='best_model.pth',
                        help='Path to save best model')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize a sample before training')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CubiCasa Wall Segmentation Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Data folder: {args.data_folder}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    # 1. Create datasets
    print("\n📁 Loading datasets...")
    train_dataset = FloorplanSVG(
        data_folder=args.data_folder,
        data_file=args.train_file,
        transform=get_train_transform()
    )
    
    val_dataset = FloorplanSVG(
        data_folder=args.data_folder,
        data_file=args.val_file,
        transform=get_val_transform()
    )
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")
    
    # Visualize a sample if requested
    if args.visualize:
        print("\n  Visualizing sample...")
        sample = train_dataset[0]
        visualize_sample(sample)
    
    # 2. Create data loaders
    print("\n Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 3. Create model
    print("\n  Creating model...")
    model = create_model(encoder_name=args.encoder)
    print(f" Model created with encoder: {args.encoder}")
    
    # 4. Create loss and optimizer
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    
    # 5. Train
    print("\n🚀 Starting training...\n")
    history = train_full_metrics(
        model=model,
        trainloader=train_loader,
        valloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=DEVICE,
        save_path=args.save_path
    )
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print(f"📊 Best model saved to: {args.save_path}")
    print("="*60)
    
    return history


if __name__ == '__main__':
    main()
