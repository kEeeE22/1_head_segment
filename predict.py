"""
Script to predict wall segmentation on a single image.

Usage:
    python predict.py --image path/to/image.png --model best_model.pth --output output.png
"""

import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from net.model import create_model
from data_class.dataset import get_val_transform
from config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, encoder_name='resnet34', num_classes=2, device=DEVICE):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path (str): Path to saved model weights
        encoder_name (str): Encoder backbone name
        num_classes (int): Number of output classes
        device (str): Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    print(f"Loading model from: {model_path}")
    model = create_model(
        encoder_name=encoder_name,
        encoder_weights=None,  # Don't load pretrained weights
        num_classes=num_classes,
        device=device
    )
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    return model


def load_and_preprocess_image(image_path, input_size=INPUT_SIZE):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to input image
        input_size (int): Target size for model input
        
    Returns:
        tuple: (preprocessed_tensor, original_image, original_size)
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    original_size = image.shape[:2]  # (height, width)
    
    # Resize to model input size
    image_resized = cv2.resize(image, (input_size, input_size))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - IMAGENET_MEAN) / IMAGENET_STD
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    
    # Add batch dimension (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_image, original_size


@torch.no_grad()
def predict(model, image_tensor, device=DEVICE):
    """
    Run prediction on preprocessed image tensor.
    
    Args:
        model (torch.nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        device (str): Device to run prediction on
        
    Returns:
        numpy.ndarray: Predicted mask (H, W) with values 0 or 1
    """
    image_tensor = image_tensor.to(device)
    
    # Forward pass
    logits = model(image_tensor)
    
    # Get predictions (argmax over classes)
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    
    # Convert to numpy and remove batch dimension
    mask = preds.cpu().numpy()[0]  # (H, W)
    
    return mask


def postprocess_mask(mask, original_size):
    """
    Resize mask back to original image size.
    
    Args:
        mask (numpy.ndarray): Predicted mask
        original_size (tuple): Original image size (height, width)
        
    Returns:
        numpy.ndarray: Resized mask
    """
    # Resize to original size
    mask_resized = cv2.resize(
        mask.astype(np.uint8), 
        (original_size[1], original_size[0]),  # (width, height)
        interpolation=cv2.INTER_NEAREST
    )
    return mask_resized


def visualize_prediction(original_image, mask, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        original_image (numpy.ndarray): Original RGB image
        mask (numpy.ndarray): Predicted mask
        save_path (str, optional): Path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Wall Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = original_image.copy()
    # Create colored mask (red for walls)
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask == 1] = [255, 0, 0]  # Red for walls
    
    # Blend with original image
    overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red = Wall)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    
    plt.show()


def save_mask(mask, save_path):
    """
    Save predicted mask as image.
    
    Args:
        mask (numpy.ndarray): Predicted mask
        save_path (str): Path to save mask
    """
    # Convert to 0-255 range
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), mask_image)
    print(f"💾 Mask saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Predict wall segmentation on a single image')
    
    # Required arguments
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to trained model weights')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output mask (default: input_name_mask.png)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization')
    parser.add_argument('--save_viz', type=str, default=None,
                        help='Path to save visualization (default: input_name_viz.png)')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone (must match training)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (must match training)')
    
    args = parser.parse_args()
    
    # Validate inputs
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Set default output paths
    if args.output is None:
        args.output = image_path.parent / f"{image_path.stem}_mask.png"
    
    if args.save_viz is None and args.visualize:
        args.save_viz = image_path.parent / f"{image_path.stem}_viz.png"
    
    print("="*60)
    print("CubiCasa Wall Segmentation - Prediction")
    print("="*60)
    print(f"Input image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # 1. Load model
    model = load_model(
        model_path=str(model_path),
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        device=DEVICE
    )
    
    # 2. Load and preprocess image
    print(f"\n📷 Loading image: {image_path}")
    image_tensor, original_image, original_size = load_and_preprocess_image(str(image_path))
    print(f"✅ Image loaded: {original_size[1]}x{original_size[0]} -> {INPUT_SIZE}x{INPUT_SIZE}")
    
    # 3. Run prediction
    print(f"\n🔮 Running prediction...")
    mask = predict(model, image_tensor, device=DEVICE)
    print(f"✅ Prediction complete")
    
    # 4. Postprocess (resize to original size)
    print(f"\n🔄 Resizing mask to original size...")
    mask_resized = postprocess_mask(mask, original_size)
    
    # Calculate statistics
    wall_pixels = np.sum(mask_resized == 1)
    total_pixels = mask_resized.size
    wall_percentage = (wall_pixels / total_pixels) * 100
    
    print(f"📊 Wall pixels: {wall_pixels:,} / {total_pixels:,} ({wall_percentage:.2f}%)")
    
    # 5. Save mask
    save_mask(mask_resized, args.output)
    
    # 6. Visualize if requested
    if args.visualize:
        print(f"\n🎨 Creating visualization...")
        visualize_prediction(
            original_image, 
            mask_resized, 
            save_path=args.save_viz
        )
    
    print("\n" + "="*60)
    print("✅ Prediction completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
