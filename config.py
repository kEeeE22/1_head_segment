"""
Configuration and constants for CubiCasa wall segmentation.
"""

import torch

# Room type mapping from CubiCasa to FUSE format
CUBICASA_TO_FUSE_ROOM = {
    0: 0,    # background
    1: 6,    # outdoor / balcony
    2: 10,   # wall
    3: 3,    # kitchen
    4: 3,    # dining / living
    5: 4,    # bedroom
    6: 2,    # bathroom
    7: 5,    # hall
    8: 5,    # railing
    9: 1,    # closet
    10: 6,   # garage
    11: 3    # generic room
}

# Model input size
INPUT_SIZE = 512

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default data paths (can be overridden)
DEFAULT_DATA_FOLDER = '/dataset/cubicasa5k/cubicasa5k/'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'
TEST_FILE = 'test.txt'

# Training hyperparameters
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EPOCHS = 100

# Image normalization (ImageNet stats)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Wall Segmentation Model configuration
ENCODER_NAME = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
NUM_CLASSES = 2  # background and wall

# DPVR Model configuration
DPVR_NUM_ROOM_CLASSES = 9
DPVR_NUM_BOUNDARY_CLASSES = 3
DPVR_ENCODER = 'vgg16'

# Logging and checkpoints
LOG_DIR = 'log/store2'
CHECKPOINT_DIR = 'checkpoint/'
