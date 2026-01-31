"""
Simple API for wall segmentation prediction.

Example usage:
    from predict_api import WallSegmentationPredictor
    
    # Initialize predictor
    predictor = WallSegmentationPredictor(model_path='best_model.pth')
    
    # Predict on single image
    mask = predictor.predict('path/to/image.png')
    
    # Predict and visualize
    mask, viz = predictor.predict_and_visualize('path/to/image.png')
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src import (
    create_model,
    DEVICE,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class WallSegmentationPredictor:
    """
    Simple API for wall segmentation prediction.
    """
    
    def __init__(self, model_path, encoder_name='resnet34', num_classes=2, 
                 device=DEVICE, input_size=INPUT_SIZE):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to trained model weights
            encoder_name (str): Encoder backbone name
            num_classes (int): Number of output classes
            device (str): Device to run on ('cuda' or 'cpu')
            input_size (int): Model input size
        """
        self.device = device
        self.input_size = input_size
        self.model = self._load_model(model_path, encoder_name, num_classes)
        
    def _load_model(self, model_path, encoder_name, num_classes):
        """Load model from checkpoint."""
        print(f"Loading model from: {model_path}")
        model = create_model(
            encoder_name=encoder_name,
            encoder_weights=None,
            num_classes=num_classes,
            device=self.device
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✅ Model loaded on {self.device}")
        return model
    
    def _preprocess(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image (numpy.ndarray or str): RGB image or path to image
            
        Returns:
            tuple: (preprocessed_tensor, original_image, original_size)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not read image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        original_image = image.copy()
        original_size = image.shape[:2]
        
        # Resize
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - IMAGENET_MEAN) / IMAGENET_STD
        
        # To tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_image, original_size
    
    def _postprocess(self, mask, original_size):
        """Resize mask to original image size."""
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        return mask_resized
    
    @torch.no_grad()
    def predict(self, image, return_original_size=True):
        """
        Predict wall segmentation on image.
        
        Args:
            image (numpy.ndarray or str): RGB image or path to image
            return_original_size (bool): If True, resize mask to original image size
            
        Returns:
            numpy.ndarray: Predicted mask (0=background, 1=wall)
        """
        # Preprocess
        image_tensor, original_image, original_size = self._preprocess(image)
        
        # Predict
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        preds = torch.argmax(logits, dim=1)
        mask = preds.cpu().numpy()[0]
        
        # Postprocess
        if return_original_size:
            mask = self._postprocess(mask, original_size)
        
        return mask
    
    def predict_with_stats(self, image):
        """
        Predict and return mask with statistics.
        
        Args:
            image (numpy.ndarray or str): RGB image or path to image
            
        Returns:
            dict: Dictionary with 'mask', 'wall_pixels', 'total_pixels', 'wall_percentage'
        """
        mask = self.predict(image)
        
        wall_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        wall_percentage = (wall_pixels / total_pixels) * 100
        
        return {
            'mask': mask,
            'wall_pixels': int(wall_pixels),
            'total_pixels': int(total_pixels),
            'wall_percentage': float(wall_percentage)
        }
    
    def predict_and_visualize(self, image, save_path=None):
        """
        Predict and create visualization.
        
        Args:
            image (numpy.ndarray or str): RGB image or path to image
            save_path (str, optional): Path to save visualization
            
        Returns:
            tuple: (mask, visualization_image)
        """
        # Load original image if path
        if isinstance(image, (str, Path)):
            original_image = cv2.imread(str(image))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image.copy()
        
        # Predict
        mask = self.predict(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Predicted Wall Mask')
        axes[1].axis('off')
        
        # Overlay
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask == 1] = [255, 0, 0]
        overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Red = Wall)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        # Convert figure to image
        fig.canvas.draw()
        viz_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        viz_image = viz_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return mask, viz_image
    
    def predict_batch(self, image_paths, batch_size=4):
        """
        Predict on multiple images.
        
        Args:
            image_paths (list): List of image paths
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of predicted masks
        """
        masks = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for path in batch_paths:
                mask = self.predict(path)
                masks.append(mask)
        
        return masks


# Example usage
if __name__ == '__main__':
    # Initialize predictor
    predictor = WallSegmentationPredictor(
        model_path='best_model.pth',
        encoder_name='resnet34'
    )
    
    # Example 1: Simple prediction
    print("Example 1: Simple prediction")
    mask = predictor.predict('path/to/image.png')
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")
    
    # Example 2: Prediction with statistics
    print("\nExample 2: Prediction with statistics")
    result = predictor.predict_with_stats('path/to/image.png')
    print(f"Wall percentage: {result['wall_percentage']:.2f}%")
    
    # Example 3: Prediction with visualization
    print("\nExample 3: Prediction with visualization")
    mask, viz = predictor.predict_and_visualize(
        'path/to/image.png',
        save_path='output_viz.png'
    )
    
    # Example 4: Batch prediction
    print("\nExample 4: Batch prediction")
    image_paths = ['image1.png', 'image2.png', 'image3.png']
    masks = predictor.predict_batch(image_paths)
    print(f"Processed {len(masks)} images")
