"""
Dataset class and data transformations for CubiCasa5k floorplans.
"""

import cv2
import numpy as np
import torch
from numpy import genfromtxt
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

# Add parent directory to path to import CubiCasa5k
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CubiCasa5k.floortrans.loaders.house import House

from config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


class FloorplanSVG(Dataset):
    """
    Dataset class for loading CubiCasa5k floorplan images and wall masks.
    
    Args:
        data_folder (str): Root folder containing the dataset
        data_file (str): Text file with list of sample folders
        transform (callable, optional): Albumentations transform to apply
        format (str): Data format, either 'txt' or 'lmdb'
        original_size (bool): Whether to use original image size
        lmdb_folder (str): LMDB folder name if using LMDB format
    """
    
    def __init__(self, data_folder, data_file, transform=None, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/'):
        self.transform = transform
        self.get_data = None
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            import lmdb
            self.lmdb = lmdb.open(
                data_folder + lmdb_folder, 
                readonly=True, 
                max_readers=8, 
                lock=False, 
                readahead=True, 
                meminit=False
            )
            self.get_data = self.get_lmdb
            self.is_transform = False
            
        self.data_folder = data_folder 
        self.folders = genfromtxt(data_folder + data_file, dtype='str')

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: Dictionary with 'image' and 'wall_mask' keys
        """
        sample = self.get_data(index)
        image = sample['image']
        mask = sample['wall_mask']
            
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            
        return {'image': image, 'wall_mask': mask}

    def get_txt(self, index):
        """
        Load sample from text file format.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Dictionary with 'image' and 'wall_mask'
        """
        fplan = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)
        
        height, width, nchannel = fplan.shape
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        raw_tensor = house.get_segmentation_tensor()
        segmentation_map = raw_tensor[0]

        # Create wall mask (wall=1, railing=1, others=0)
        mask = np.zeros((height, width), dtype=np.float32)
        mask[segmentation_map == 2] = 1.0  # wall
        mask[segmentation_map == 8] = 1.0  # railing
        
        sample = {
            'image': fplan.astype(np.uint8),
            'wall_mask': mask  
        }
        return sample

    def get_lmdb(self, index):
        """Load sample from LMDB format (placeholder)."""
        raise NotImplementedError("LMDB loading not implemented yet")


def get_train_transform(input_size=INPUT_SIZE):
    """
    Get training data augmentation pipeline.
    
    Args:
        input_size (int): Target image size
        
    Returns:
        albumentations.Compose: Transform pipeline
    """
    return A.Compose([
        # 1. Resize to fixed input size
        A.Resize(input_size, input_size),

        # 2. Geometric transformations (safe for CAD drawings)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),

        # 3. Pixel-level transformations (simulate scan quality)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

        # 4. Normalization (required)
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transform(input_size=INPUT_SIZE):
    """
    Get validation/test data transform pipeline.
    
    Args:
        input_size (int): Target image size
        
    Returns:
        albumentations.Compose: Transform pipeline
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
