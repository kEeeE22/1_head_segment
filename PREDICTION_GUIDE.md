# Wall Segmentation Prediction Guide

This guide explains how to use the prediction scripts to predict wall segmentation on single images.

## 📋 Prerequisites

1. **Trained Model**: You need a trained model file (e.g., `best_model.pth`)
   - Train a model using: `python train_model.py --data_folder /path/to/data`
   
2. **Dependencies**: Make sure all requirements are installed
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Method 1: Command Line Interface (CLI)

The simplest way to predict on a single image:

```bash
python predict.py --image path/to/image.png --model best_model.pth --visualize
```

**Arguments:**
- `--image`: Path to input image (required)
- `--model`: Path to trained model weights (default: `best_model.pth`)
- `--output`: Path to save output mask (default: `input_name_mask.png`)
- `--visualize`: Show visualization window
- `--save_viz`: Path to save visualization (default: `input_name_viz.png`)
- `--encoder`: Encoder backbone, must match training (default: `resnet34`)
- `--num_classes`: Number of classes, must match training (default: `2`)

**Example:**
```bash
# Basic prediction
python predict.py --image floorplan.png --model best_model.pth

# With visualization
python predict.py --image floorplan.png --model best_model.pth --visualize

# Custom output paths
python predict.py --image floorplan.png --model best_model.pth \
    --output my_mask.png --save_viz my_viz.png
```

### Method 2: Python API

For programmatic use in your own scripts:

```python
from predict_api import WallSegmentationPredictor

# Initialize predictor
predictor = WallSegmentationPredictor(model_path='best_model.pth')

# Predict on single image
mask = predictor.predict('path/to/image.png')

# Predict with statistics
result = predictor.predict_with_stats('path/to/image.png')
print(f"Wall percentage: {result['wall_percentage']:.2f}%")

# Predict with visualization
mask, viz = predictor.predict_and_visualize(
    'path/to/image.png',
    save_path='output_viz.png'
)
```

## 📚 API Reference

### WallSegmentationPredictor Class

#### Initialization
```python
predictor = WallSegmentationPredictor(
    model_path='best_model.pth',    # Path to trained model
    encoder_name='resnet34',         # Encoder backbone
    num_classes=2,                   # Number of classes
    device='cuda',                   # 'cuda' or 'cpu'
    input_size=512                   # Model input size
)
```

#### Methods

**1. `predict(image, return_original_size=True)`**
- Predict wall segmentation on image
- **Args:**
  - `image`: numpy array (RGB) or path to image
  - `return_original_size`: If True, resize mask to original image size
- **Returns:** numpy array with predicted mask (0=background, 1=wall)

**2. `predict_with_stats(image)`**
- Predict and return statistics
- **Returns:** Dictionary with:
  - `mask`: Predicted mask
  - `wall_pixels`: Number of wall pixels
  - `total_pixels`: Total number of pixels
  - `wall_percentage`: Percentage of wall pixels

**3. `predict_and_visualize(image, save_path=None)`**
- Predict and create visualization
- **Returns:** Tuple of (mask, visualization_image)

**4. `predict_batch(image_paths, batch_size=4)`**
- Predict on multiple images
- **Args:**
  - `image_paths`: List of image paths
  - `batch_size`: Batch size for processing
- **Returns:** List of predicted masks

## 💡 Examples

### Example 1: Simple Prediction
```python
from predict_api import WallSegmentationPredictor
import numpy as np

predictor = WallSegmentationPredictor(model_path='best_model.pth')
mask = predictor.predict('floorplan.png')

print(f"Mask shape: {mask.shape}")
print(f"Unique values: {np.unique(mask)}")
```

### Example 2: Get Statistics
```python
predictor = WallSegmentationPredictor(model_path='best_model.pth')
result = predictor.predict_with_stats('floorplan.png')

print(f"Wall pixels: {result['wall_pixels']:,}")
print(f"Total pixels: {result['total_pixels']:,}")
print(f"Wall percentage: {result['wall_percentage']:.2f}%")
```

### Example 3: Save Mask as Image
```python
import cv2
from predict_api import WallSegmentationPredictor

predictor = WallSegmentationPredictor(model_path='best_model.pth')
mask = predictor.predict('floorplan.png')

# Save as binary image (0=black, 255=white)
mask_image = (mask * 255).astype(np.uint8)
cv2.imwrite('wall_mask.png', mask_image)
```

### Example 4: Batch Processing
```python
from predict_api import WallSegmentationPredictor

predictor = WallSegmentationPredictor(model_path='best_model.pth')

# Process multiple images
image_paths = ['floor1.png', 'floor2.png', 'floor3.png']
masks = predictor.predict_batch(image_paths, batch_size=2)

for i, mask in enumerate(masks):
    wall_pct = (np.sum(mask == 1) / mask.size) * 100
    print(f"Image {i+1}: {wall_pct:.2f}% walls")
```

### Example 5: Process from Numpy Array
```python
import cv2
from predict_api import WallSegmentationPredictor

# Load image as numpy array
image = cv2.imread('floorplan.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict
predictor = WallSegmentationPredictor(model_path='best_model.pth')
mask = predictor.predict(image)
```

## 📊 Output Format

### Mask Values
- `0`: Background (non-wall)
- `1`: Wall

### Mask Properties
- **Type**: numpy.ndarray
- **Shape**: (height, width)
- **Dtype**: uint8
- **Values**: 0 or 1

## 🎨 Visualization

The visualization shows three panels:
1. **Original Image**: Input floorplan image
2. **Predicted Mask**: Binary mask (white=wall, black=background)
3. **Overlay**: Original image with red overlay on detected walls

## ⚙️ Advanced Usage

### Custom Model Configuration
```python
# Use different encoder
predictor = WallSegmentationPredictor(
    model_path='best_model.pth',
    encoder_name='resnet50',  # Different encoder
    num_classes=2
)
```

### CPU-only Inference
```python
# Force CPU usage
predictor = WallSegmentationPredictor(
    model_path='best_model.pth',
    device='cpu'
)
```

### Custom Input Size
```python
# Use different input size (must match training)
predictor = WallSegmentationPredictor(
    model_path='best_model.pth',
    input_size=256  # Smaller input size
)
```

## 🐛 Troubleshooting

### Issue: Model file not found
```
FileNotFoundError: Model not found: best_model.pth
```
**Solution:** Make sure you have trained a model first:
```bash
python train_model.py --data_folder /path/to/data --epochs 100
```

### Issue: Image not found
```
FileNotFoundError: Image not found: image.png
```
**Solution:** Check the image path is correct and the file exists.

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Use CPU instead:
```bash
python predict.py --image image.png --model best_model.pth
# The script will automatically use CPU if CUDA is not available
```

Or in Python:
```python
predictor = WallSegmentationPredictor(model_path='best_model.pth', device='cpu')
```

### Issue: Model architecture mismatch
```
RuntimeError: Error(s) in loading state_dict
```
**Solution:** Make sure encoder and num_classes match the training configuration:
```bash
python predict.py --image image.png --model best_model.pth \
    --encoder resnet34 --num_classes 2
```

## 📝 Notes

- The model expects RGB images
- Images are automatically resized to the model's input size (default: 512x512)
- Output masks are resized back to the original image size
- The prediction uses the same normalization as training (ImageNet stats)

## 🔗 Related Files

- `predict.py`: Command-line prediction script
- `predict_api.py`: Python API for programmatic use
- `example_predict.py`: Example usage scripts
- `train_model.py`: Training script
- `src/`: Source code modules
